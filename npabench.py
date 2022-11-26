# Copyright (c) 2022 O. Masoud

import os
import sys
import gc
import functools
import numpy as np
import pickle
import h5py
import tables
import zarr
from timeit import default_timer as timer
import datetime as dt
import tempfile
from pathlib import Path
import shutil
import argparse
import traceback
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker
import cpuinfo
import re
import webbrowser
from io import BytesIO
import base64
import subprocess
import json
import psutil
import distro
import platform
import html
import struct
from typing import Callable
import numpy.typing as npt

def get_powershell_output_object(cmd):
	result = subprocess.run(['powershell.exe', '-NonInteractive', '-NoProfile',  '-Command', cmd], capture_output=True) # Assuming not needed: '-ExecutionPolicy', 'Unrestricted'
	if (result.stderr):
		s_err=result.stderr.decode('utf8',errors='ignore')
		raise RuntimeError(s_err)
	out = result.stdout.decode('utf8',errors='ignore')
	if not out: # empty output
		return None
	return json.loads(out)

def get_disk_info():
	ret = 'No disk info'

	with tempfile.NamedTemporaryFile('wb') as f:
		filepath = f.name # we just need a name; file will be deleted

	drive = os.path.splitdrive(filepath)[0]
	if not drive or len(drive)!=2 or drive[1]!=':': # no drive present
		return ret

	cmd=r'''
Get-Disk | ForEach-Object { 
	$disk=$_
	$disk | 
		Get-Partition | 
		Where-Object DriveLetter -eq '<DRIVELETTER>' | 
		Select-Object DriveLetter, @{n='Type';e={ $disk.BusType }}, @{n='Model';e={ $disk.Model }}
	} | 
	ConvertTo-Json
'''

	cmd = cmd.replace('<DRIVELETTER>',drive[0]) # the first character only
	try:
		info = get_powershell_output_object(cmd)
	except:
		info = {}

	if info and info is not None:
		ret = f'drive: {info["DriveLetter"]}; type: {info["Type"]}; model: {info["Model"]}'

	return ret


def get_os_info():
	ret = f'{platform.system()} {platform.release()} ({platform.version()})'
	info = [distro.name(), distro.version(), distro.codename()]
	if any(info):
		ret += ': ' + '-'.join(info)
	return ret

def elapsed(reset=True):
	now = timer()
	try:
		return now - elapsed.last_timestamp
	except:
		reset=True
		return None
	finally:
		if reset:
			elapsed.last_timestamp = now

class StopWatch():
	def __init__(self) -> None:
		self.last_timestamp = timer()
	def elapsed(self, reset=True):
		now = timer()
		diff = now - self.last_timestamp
		if reset:
			self.last_timestamp = now
		return diff
	


def print_on_same_line(*objects,sep=' ',file=sys.stdout):
	class A:
		def __init__(self):
			self.previous_string_length=0
			self.string=''
		def write(self,string):
			#print(f'received "{string}", {[ord(c) for c in string]}')
			self.string+=string
		def get_and_reset(self):
			string_length=len(self.string)
			erase_padding=''
			if self.previous_string_length>string_length:
				erase_padding=' '*(self.previous_string_length-string_length)
			output=self.string+erase_padding
			self.previous_string_length=string_length
			self.string=''
			return output	 
	try:
		print_on_same_line.writer
	except:
		print_on_same_line.writer=A()
		
	print(*objects, sep=sep, file=print_on_same_line.writer,end='')
	print(print_on_same_line.writer.get_and_reset(), end='\r', flush=True, file=file)


def profile_write(fmt, arr):
	gc.disable()
	elapsed()
	format_rw[fmt][1](arr)
	e = elapsed()
	gc.enable()
	return e

def profile_read(fmt, arr=None):
	gc.disable()
	elapsed()
	a=format_rw[fmt][2]()
	e = elapsed()
	gc.enable()
	if arr is not None:
		assert np.array_equal(a,arr)
	return e


def pickle_w(fpath,a):
	with open(fpath,'wb') as f:
		pickle.dump(a,f)
def pickle_r(fpath):
	with open(fpath,'rb') as f:
		return pickle.load(f)
def hdf5_w(fpath,a):
	with h5py.File(fpath, 'w') as f:
		f.create_dataset('data',data=a)
def hdf5_r(fpath):
	with h5py.File(fpath, "r") as f:
		return f["data"][()]  # type: ignore
def pytables_w(fpath,a):
	with tables.open_file(fpath, 'w') as f:
		gcolumns = f.create_group(f.root, 'columns', 'data')
		f.create_array(gcolumns, 'data', a, 'data')
def pytables_r(fpath):
	with tables.open_file(fpath, 'r') as f:
		return f.root.columns.data[()]

# from https://github.com/divideconcept/fastnumpyio/blob/main/fastnumpyio.py
def fast_numpy_save(array):
	size=len(array.shape)
	if size>255:
		raise ValueError('Dimensions greater than 255 are not supported')
	return bytes(array.dtype.byteorder.replace('=','<' if sys.byteorder == 'little' else '>')+ 
				array.dtype.kind,
				'utf-8')+ \
			array.dtype.itemsize.to_bytes(1,byteorder='little')+ \
			struct.pack(f'<B{size}Q',size,*array.shape)+ \
			array.tobytes()

def fast_numpy_load(data):
	dtype = str(data[:2],'utf-8')
	dtype += str(data[2])
	size = data[3]
	shape = struct.unpack_from(f'<{size}Q', data, 4)
	return np.ndarray(shape, dtype=dtype, buffer=data[4+size*8:])


fpath = '' # to be set outside

format_rw = {
	'np': ('.npy',lambda a: np.save(fpath,a), lambda: np.load(fpath)), # extension needed otherwise will append .npy 
	'npz': ('.npz',lambda a: np.savez(fpath,a), lambda: np.load(fpath)['arr_0']), # extension needed otherwise will append .npz 
	'npzc': ('.npz',lambda a: np.savez_compressed(fpath,a), lambda: np.load(fpath)['arr_0']), # extension needed otherwise will append .npz 
	'hdf5': ('.hdf5',lambda a: hdf5_w(fpath,a), lambda: hdf5_r(fpath)), # give it .hdf5 extension (optional)
	'pickle': ('.pkl',lambda a: pickle_w(fpath,a), lambda: pickle_r(fpath)), # give it .pkl extension (optional)
	'zarr_zip': ('.zip',lambda a: zarr.save_array(fpath,a), lambda: zarr.load(fpath)), # there's zip and zarr; need to say zip
	'zarr': ('.zarr',lambda a: zarr.save_array(fpath,a), lambda: zarr.load(fpath)), # there's zip and zarr; need to say zarr (which gets stored as a directory)
	'pytables': ('.h5',lambda a: pytables_w(fpath,a), lambda: pytables_r(fpath)), # give it .h5 extension (optional)
	'fast_np': ('.fnp',lambda a: open(fpath,'wb').write(fast_numpy_save(a)), lambda: fast_numpy_load(open(fpath,'rb').read())), # give it .fnp extension (optional)
}		

def get_outlier_mask(a, axis=-1, thresh=2.0):
	d = np.abs(a - np.median(a, axis=axis, keepdims=True))
	mdev = np.median(d, axis=axis, keepdims=True)
	mdev[mdev<=np.finfo(a.dtype).eps]=1.0
	s = d / mdev
	return s>thresh

def relabel_size_axis(ax, max_pwr, start_pwr=0):
	ax.set_xticks([2**k for k in range(start_pwr,max_pwr+1)])
	ax.set_xticklabels([
		'1Byte','2','4','8','16','32','64','128','256','\u00BD'+'KB',
		'1','2','4','8KB','16','32','64','128','256','\u00BD'+'MB',
		'1','2','4MB','8','16','32','64','128','256','\u00BD'+'GB',
		'1','2GB','4','8','16','32','64','128','256','\u00BD'+'TB',
		'1','2','4','8','16','32','64','128','256','\u00BD'+'PB'][start_pwr:max_pwr+1],fontsize=8)
	ax.tick_params(labelright=True, right=True)

def relabel_time_axis(ax, yres=1.0):
	ax.tick_params(labelright=True, right=True)
	seconds=np.power(10.0,np.arange(-4,4,yres))
	ax.set_yticks(seconds)
	ax.set_yticklabels([f'{s:.4f}' if s<.001 else (f'{s:.3f}' if s<10 else f'{s:.0f}') for s in seconds],fontsize=8,fontname ='Times New Roman')	


def summary_plot_io_time(accum, data_dist, wr_rd, title):
	data_dist = data_dist if isinstance(data_dist, tuple) else (data_dist,)
	max_pwr = accum.shape[1]-1
	y=np.nanmean(accum,axis=-1)[data_dist,:,:,wr_rd].mean(axis=0) # shape (max_pwr,8)
	y_no_outliers = np.ma.array(y,mask=get_outlier_mask(np.log(y),axis=1,thresh=9.0)) # outliers decided in the log domain
	y_min = y_no_outliers.min(axis=1)
	y_max = y_no_outliers.max(axis=1)
	x=np.broadcast_to(np.array([2**k for k in range(max_pwr+1)])[:,np.newaxis],y.shape)

	fig=plt.figure(figsize = (12, 8))
	fig.subplots_adjust(
		top=0.95,
		bottom=0.05,
		left=0.05,
		right=0.95,
		hspace=0.2,
		wspace=0.25
	)
	fig.suptitle(title, fontsize=14)

	ax1=plt.subplot(211)

	ax1.loglog(x,y)
	relabel_size_axis(ax1, max_pwr=max_pwr)
	relabel_time_axis(ax1)
	ax1.set_xlim(left=1, right=2**max_pwr)
	ax1.set_ylim(bottom=y[0,:].min(), top=y[max_pwr,:].max())
	#xaxis/yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()) unlikely needed
	ax1.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
	ax1.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
	ax1.set_xlabel('Size',loc='right')
	ax1.set_ylabel('Sec',loc='top')
	ax1.legend(list(format_rw.keys()))

	def zoomed_plot(a,b,ax,yres):
		ax.loglog(x,y)
		relabel_size_axis(ax, max_pwr=max_pwr)
		relabel_time_axis(ax, yres=yres)
		ax.set_xlim(left=2**a, right=2**b)
		ax.set_ylim(bottom=min(y_min[a],y_min[b]), top=max(y_max[a],y_max[b]))
		ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
		ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
		if y_no_outliers[a:b+1,:].mask.any():
			plt.text(0.0, 1.01, 'At least one outlier not shown', fontsize=8, transform=ax.transAxes)

	def make_zoomed_plot(prev,a_size_str,b_size_str,subplot_value,yres):
		if max_pwr>prev:
			a,b = size_str_to_pwr(a_size_str), size_str_to_pwr(b_size_str)
			if max_pwr<b:
				a,b = max(0, max_pwr - (b-a)), max_pwr
			zoomed_plot(a,b,plt.subplot(subplot_value),yres=yres)
			prev=b
		return prev

	prev = make_zoomed_plot(0,'8kb','256kb',234,0.15)
	prev = make_zoomed_plot(prev,'4mb','16mb',235,0.25)
	prev = make_zoomed_plot(prev,'2gb','4gb',236,0.1)

	return fig

def summary_plot_io_rate(accum):
	data_dist=(0,1)
	max_pwr = accum.shape[1]-1
	y1=np.nanmean(accum,axis=-1)[data_dist,:,:,0].mean(axis=0) # write time, all arrays
	y2=np.nanmean(accum,axis=-1)[data_dist,:,:,1].mean(axis=0) # read time, all arrays

	fig=plt.figure(figsize = (12, 8))
	fig.subplots_adjust(
		top=0.90,
		bottom=0.05,
		left=0.05,
		right=0.95,
		hspace=0.2,
		wspace=0.25
	)
	x=np.broadcast_to(np.array([2**k for k in range(max_pwr+1)])[:,np.newaxis],y1.shape) # y1 and y2 have the same shape
	start_pwr=size_str_to_pwr('8kb')
	if max_pwr<=size_str_to_pwr('1mb'):
		start_pwr=0


	ax1=plt.subplot(211)
	ax1.loglog(x,x/y1/1024/1024/1024)
	ax1.set_yscale('linear')
	relabel_size_axis(ax1, max_pwr=max_pwr, start_pwr=start_pwr)
	ax1.set_xlim(left=2**start_pwr, right=2**max_pwr)
	ax1.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
	ax1.grid(axis='y')
	ax1.set_ylabel('GB/Sec',loc='top')
	ax1.set_title(r'write rate')
	ax1.legend(list(format_rw.keys()))


	ax2=plt.subplot(212)
	ax2.loglog(x,x/y2/1024/1024/1024)
	ax2.set_yscale('linear')
	relabel_size_axis(ax2, max_pwr=max_pwr, start_pwr=start_pwr)
	ax2.set_xlim(left=2**start_pwr, right=2**max_pwr)
	ax2.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
	#ax2.set_yticks(np.arange(0,5.5,.5))
	ax2.grid(axis='y')
	ax2.set_xlabel('Size',loc='right')
	ax2.set_title(r'read rate')
	fig.suptitle('IO Rate', fontsize=14)

	return fig

def summary_plot_file_size(file_size):
	max_pwr = file_size.shape[1]-1
	y1=file_size[0,:,:]
	y2=file_size[1,:,:]

	fig=plt.figure(figsize = (12, 8))
	fig.subplots_adjust(
		top=0.90,
		bottom=0.05,
		left=0.05,
		right=0.95,
		hspace=0.2,
		wspace=0.25
	)

	ax1=plt.subplot(211)
	x=np.broadcast_to(np.array([2**k for k in range(max_pwr+1)])[:,np.newaxis],y1.shape) # y1 and y2 have the same shape
	ax1.loglog(x,x/y1)
	ax1.set_yscale('linear')
	relabel_size_axis(ax1, max_pwr)
	ax1.set_xlim(left=1, right=2**max_pwr)
	ax1.grid(axis='y')
	ax1.set_ylabel('Compression Ratio',loc='top')
	ax1.legend(list(format_rw.keys()))
	ax1.set_title('random numbers')

	ax2=plt.subplot(212)
	ax2.loglog(x,x/y2)
	ax2.set_yscale('linear')
	relabel_size_axis(ax2, max_pwr)
	ax2.set_xlim(left=1, right=2**max_pwr)
	ax2.grid(axis='y')
	ax2.set_xlabel('Size',loc='right')
	ax2.set_title(r'80% sparse random numbers')

	fig.suptitle('File Size', fontsize=14)

	return fig

def get_lib_version_info():
	return {
		'numpy' : f'{np.version.version}',
		'pickle' : f'{pickle.format_version}',  # type: ignore
		'h5py' : f'{h5py.version.version} (using hdf5 version: {h5py.version.hdf5_version})',
		'tables' : f'{tables.get_pytables_version()} (using hdf5 version: {tables.hdf5_version})',
		'zarr' : f'{zarr.__version__}'
	}


@functools.lru_cache(maxsize=1) # no need to repopulate
def get_sys_info():
	return {
		'Processor' : cpuinfo.get_cpu_info()['brand_raw'],
		'Disk' : get_disk_info(),
		'Memory' : pwr_to_size_str(round(np.log2(psutil.virtual_memory().total))),
		'OS' : get_os_info()
	}

def view_on_github_svg():
	return r'''
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="122" height="20">
	<linearGradient id="b" x2="0" y2="100%">
		<stop offset="0" stop-color="#bbb" stop-opacity=".1" />
		<stop offset="1" stop-opacity=".1" />
	</linearGradient>
	<clipPath id="a">
		<rect width="122" height="20" rx="3" fill="#fff" />
	</clipPath>
	<g clip-path="url(#a)">
		<path fill="#2f363d" d="M0 0h25v20H0z" />
		<path fill="#586069" d="M25 0h97v20H25z" />
		<path fill="url(#b)" d="M0 0h122v20H0z" />
	</g>
	<g fill="#fff" text-anchor="middle" font-family="DejaVu Sans,Verdana,Geneva,sans-serif" font-size="110">
		<image x="5" y="3" width="14" height="14"
			xlink:href="data:image/svg+xml;base64,PHN2ZyBmaWxsPSJ3aGl0ZXNtb2tlIiByb2xlPSJpbWciIHZpZXdCb3g9IjAgMCAyNCAyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48dGl0bGU+R2l0SHViIGljb248L3RpdGxlPjxwYXRoIGQ9Ik0xMiAuMjk3Yy02LjYzIDAtMTIgNS4zNzMtMTIgMTIgMCA1LjMwMyAzLjQzOCA5LjggOC4yMDUgMTEuMzg1LjYuMTEzLjgyLS4yNTguODItLjU3NyAwLS4yODUtLjAxLTEuMDQtLjAxNS0yLjA0LTMuMzM4LjcyNC00LjA0Mi0xLjYxLTQuMDQyLTEuNjFDNC40MjIgMTguMDcgMy42MzMgMTcuNyAzLjYzMyAxNy43Yy0xLjA4Ny0uNzQ0LjA4NC0uNzI5LjA4NC0uNzI5IDEuMjA1LjA4NCAxLjgzOCAxLjIzNiAxLjgzOCAxLjIzNiAxLjA3IDEuODM1IDIuODA5IDEuMzA1IDMuNDk1Ljk5OC4xMDgtLjc3Ni40MTctMS4zMDUuNzYtMS42MDUtMi42NjUtLjMtNS40NjYtMS4zMzItNS40NjYtNS45MyAwLTEuMzEuNDY1LTIuMzggMS4yMzUtMy4yMi0uMTM1LS4zMDMtLjU0LTEuNTIzLjEwNS0zLjE3NiAwIDAgMS4wMDUtLjMyMiAzLjMgMS4yMy45Ni0uMjY3IDEuOTgtLjM5OSAzLS40MDUgMS4wMi4wMDYgMi4wNC4xMzggMyAuNDA1IDIuMjgtMS41NTIgMy4yODUtMS4yMyAzLjI4NS0xLjIzLjY0NSAxLjY1My4yNCAyLjg3My4xMiAzLjE3Ni43NjUuODQgMS4yMyAxLjkxIDEuMjMgMy4yMiAwIDQuNjEtMi44MDUgNS42MjUtNS40NzUgNS45Mi40Mi4zNi44MSAxLjA5Ni44MSAyLjIyIDAgMS42MDYtLjAxNSAyLjg5Ni0uMDE1IDMuMjg2IDAgLjMxNS4yMS42OS44MjUuNTdDMjAuNTY1IDIyLjA5MiAyNCAxNy41OTIgMjQgMTIuMjk3YzAtNi42MjctNS4zNzMtMTItMTItMTIiLz48L3N2Zz4=" />
		<text x="735" y="150" fill="#010101" fill-opacity=".3" transform="scale(.1)" textLength="870">View On
			GitHub</text><text x="735" y="140" transform="scale(.1)" textLength="870">View On GitHub</text>
	</g>
</svg>	
'''

def pretty_info_str(info, title, as_html):
	s=''
	tab=''
	if as_html: s += '<div class="monospace">\n'
	s+=f'{title}:\n'
	if as_html: s += '<table>'
	for k,v in info.items():
		if as_html: s+='<tr><td>'
		s+=f'{k+":":16}'
		if as_html: s+='</td><td>'
		s+=f'{v}\n'
		if as_html: s += '</td></tr>\n'
	if as_html: s += '</table></div>\n'
	return s	


def pretty_lib_version_info_str(libinfo, as_html=False):
	return pretty_info_str(libinfo, title='Library versions', as_html=as_html)

def pretty_sys_info_str(sysinfo, as_html=False):
	return pretty_info_str(sysinfo, title='System information', as_html=as_html)


# The instance creation time determines when the clock starts ticking
class Progress():	
	# progress_stream is an iterable that defines the entirety of the steps
	# 'ema_rem': exponential moving average of remaining time with alpha set to progress squared
	# 'ema_past_rem': exponential moving average of remaining time which is calculated from
	#                 an exponential moving average of per-unit past time, with alpha = 0.05
	def __init__(self, progress_stream, reporting_interval=0.5, method='ema_rem') -> None:
		self._step_sw = StopWatch()
		self._reporting_interval = reporting_interval
		self._progress_and_time = zip(progress_stream, Progress._time_stream())
		self._t_remaining_smooth = None
		self._per_unit_smooth = None
		self._method=method
		if self._method not in ['ema_rem', 'ema_past_rem']:
			raise ValueError(f'Unrecognized method: {self._method}.')
		
	@staticmethod
	def _time_stream(): 
		stopwatch=StopWatch()
		while True:
			yield stopwatch.elapsed(reset=False)

	def register_progress(self, report_consumer : Callable=lambda _:None):
		self._update()
		if self._step_sw.elapsed(reset=False) > self._reporting_interval: #update interval
			self._step_sw.elapsed() # reset	
			report_consumer(self._progress_str)

	@property
	def _progress_str(self):
		assert self._t_remaining_smooth is not None # must call _update first (this also avoid pylance warning)
		return (f'{self._progress_value:7.2%}'
				f'\t\tElapsed: {dt.timedelta(seconds=round(self._t_elapsed))}'
				f'\tRemaining: {dt.timedelta(seconds=round(self._t_remaining_smooth))}'
				)		
		
	def _update(self):
		self._progress_value, self._t_elapsed = next(self._progress_and_time)
		p=self._progress_value # shorthand 
		per_unit = self._t_elapsed/p

		if self._method=='ema_past_rem':
			alpha=p * .05
			if self._per_unit_smooth is None:
				self._per_unit_smooth = per_unit
			self._per_unit_smooth = (1-alpha)*self._per_unit_smooth + alpha*per_unit
			t_remaining = self._per_unit_smooth*(1-p)
		else:
			alpha = p*p
			t_remaining = per_unit*(1-p)	

		if self._t_remaining_smooth is None:
			self._t_remaining_smooth=t_remaining

		self._t_remaining_smooth = (1-alpha)*self._t_remaining_smooth + alpha*t_remaining



def size_str_to_pwr(s):
	def is_power_of_two(n):
		return (n != 0) and (n & (n-1) == 0)
	match = re.fullmatch(r'([0-9]+)([K|M|G|T|P]?B)',s.upper())
	if match is None:
		raise ValueError(f'Size needs to be something like 32MB, 1gb, 256KB, 2048B. A power of two number followed by a unit. Given input was "{s}".')
	num_part = int(match.group(1))
	if not is_power_of_two(num_part):
		raise ValueError(f'The numbers needs to be a power of 2. Given input was {s} with number={num_part}.')
	unit_pwr = {'B':0, 'KB':10, 'MB':20, 'GB':30, 'TB':40, 'PB':50}[match.group(2)] # will for sure match due to regex

	return unit_pwr + num_part.bit_length()-1

def pwr_to_size_str(pwr):
	if pwr<0 or pwr>59:
		raise ValueError(f'Power {pwr} is out of range.')
	unit_idx = pwr//10
	unit=['B','KB','MB','GB','TB','PB'][unit_idx]
	return f'{2**(pwr-unit_idx*10)}{unit}'

def add_html_header(s):
	HTML_PRE = '''
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Numpy Array I/O Benchmark Summary Report</title>
<style>
.monospace {
  font-family: monospace;
}
table td:first-child {
  width: 8em;
  min-width: 8em;
  max-width: 8em;
}
</style>
</head>
<body>	
'''

	HTML_POST = f'''
<br>
<div class="monospace" style="font-size:11px;">
<br>
<a href="https://github.com/omasoud/np-array-io-bench">
{view_on_github_svg()}
</a>
<br>
Copyright (c) 2022 O. Masoud
<br>
<br>
</div>
</body>
</html>
'''	
	return HTML_PRE+s+HTML_POST

def display_html_in_tab(s, append_headers=True):
	if append_headers:
		html_str=add_html_header(s)
	else:
		html_str=s

	with tempfile.NamedTemporaryFile('w', delete=False, suffix='.html', encoding='utf-8') as f:
		url = 'file://' + f.name
		f.write(html_str)
	webbrowser.open(url)


if __name__ == '__main__':

	print('Numpy Array File I/O Benchmark. By O. Masoud.\n')

	parser = argparse.ArgumentParser(description='Benchmark load/save speeds and summarize results graphically. A results file is generated and saved. '
												'The tool can be also run just to read a results file and show the results graphically using the '
												'--summarize-file argument.')
	group = parser.add_mutually_exclusive_group(required=False)
	group.add_argument('-s','--summarize-file', action='store', metavar='FILE',
					 help='Only generate summary graphs based on provided results file. This will not run the benchmark.')
	group.add_argument('--max-size', action='store', default='1MB', 
					help='Maximum file size (must be a power of 2) to benchmark (e.g, 8MB, 4GB, 128KB). Default: 1MB. ' 
					'Caution: large sizes can take a very long time or run out of memory or disk space. 16GB '
					'takes about 90 minutes on a fast computer.')
	parser.add_argument('--no-browser', action='store_true', help='Do not launch a browser tab to display the results.')
	parser.add_argument('--save-html-file', action='store', metavar='FILE',  help='If desired, provide filename so that html report gets saved to it.')
	parser.add_argument('--standalone-html', action='store_true', 
					help='By default the html will reference generated png files for the figures. '
					'But if desired this option can encode the pngs directly in the html (making it larger but standalone).')
	parser.add_argument('--notebook', action='store_true', help='Plot results in Jupyter Notebook.')

	args = parser.parse_args()
	if args.standalone_html and args.save_html_file is None:
		parser.error('Expecting --save-html-file when using --standalone-html')

	try:
		if not args.summarize_file:
			time_str = dt.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
			result_filepath = os.path.abspath(f'numpy_array_bench_{time_str}.pkl')
			print(f'Results will be written to: {result_filepath}\n')
			libinfo = get_lib_version_info()
			print(pretty_lib_version_info_str(libinfo))
			sysinfo = get_sys_info()
			print(pretty_sys_info_str(sysinfo))
			cmd_str = ' '.join([os.path.basename(sys.argv[0])]+sys.argv[1:])

			#MAX_PWR=32 # 32 for 4 gig; 34 for 16 gig
			#MAX_PWR=34
			MAX_PWR = size_str_to_pwr(args.max_size)

			MAX_SIZE : int = 2**MAX_PWR

			print(f'Will benchmark numpy arrays with size up to {pwr_to_size_str(MAX_PWR)}.\n')
			print('Preparing arrays...')
			dtype = np.float32
			uni : npt.NDArray[dtype] = np.random.default_rng().random(size=MAX_SIZE//dtype().itemsize, dtype=dtype)
			arrays={
				'uniform': uni,
				'sparse': uni*(uni<.2)*4 # 80% sparse, rescale back to [0,1)
			}

			def file_or_dir_size(p):
				po=Path(p)
				if po.is_dir():
					return sum(p.stat().st_size for p in po.rglob('*'))
				return po.stat().st_size

			fpath_cache = {}
			def get_fpath(fmt):
				if fmt not in fpath_cache:
					with tempfile.NamedTemporaryFile('wb', suffix=format_rw[fmt][0]) as f:
						fpath_cache[fmt]=f.name
				return fpath_cache[fmt]

			def calc_reps(size_pwr):
				return min(max(round(1.5**(33-size_pwr)),2),100)

			# return values 0.0 to 1.0 corresponding to the progress achieved by successive inner loops of the main benchmark 
			def iteration_progress():
				FUDGE = float(2**19) # adding this, representing a constant overhead, reduces the imbalance in progress rate of change
				bytes_per_inner_iter=[]
				_ = [[[[[(lambda v:bytes_per_inner_iter.append(v))(float(2**size_pwr) + FUDGE)
									for _ in range(2)]							# 1 write and 1 read
									for _ in range(calc_reps(size_pwr))] 		# reps
									for _ in range(len(format_rw))]				# formats
									for size_pwr in range(MAX_PWR+1)]			# MAX_PWR+1
									for _ in range(len(arrays))]				# arrays
				a=np.array(bytes_per_inner_iter)
				yield from np.cumsum(a)/a.sum()

			print('Running benchmark...')
			progress = Progress(progress_stream=iteration_progress(), reporting_interval=0.2, method='ema_rem')

			accum = np.full((len(arrays), MAX_PWR+1, len(format_rw), 2, calc_reps(0)),np.nan) # 2 for write&read, 10 for max reps
			file_size = np.full((len(arrays), MAX_PWR+1, len(format_rw)),np.nan)
			for ddist,arr in enumerate(arrays.values()):
				for size_pwr in range(MAX_PWR+1): # 1 byte to 4GB or whatever
					sized_arr = arr[:2**size_pwr//dtype().itemsize]
					total_reps = calc_reps(size_pwr)	
#					print(ddist,size_pwr,total_reps)
					for j,fmt in enumerate(format_rw.keys()):
						fpath=get_fpath(fmt) # reuse the same file path for each format
						for rep in range(total_reps): # some reasonable reps
							
							accum[ddist,size_pwr,j,0,rep] = profile_write(fmt, sized_arr)
							progress.register_progress(report_consumer=print_on_same_line)

							file_size[ddist,size_pwr,j] = file_or_dir_size(fpath)

							accum[ddist,size_pwr,j,1,rep] = profile_read(fmt, sized_arr if rep==0 else None) # validate only on first read
							progress.register_progress(report_consumer=print_on_same_line)

			print_on_same_line()

			print('Saving results...')
			with open(result_filepath,'wb') as f:
				pickle.dump([accum,
							file_size,
							libinfo,
							sysinfo,
							cmd_str],f)


			# Delete temps 
			print('Cleaning up...')
			for fpath in fpath_cache.values():
				po=Path(fpath)
				if po.is_dir(): # one of the zarr formats is a directory
					shutil.rmtree(po) # maybe replace with po.rmtree() after python 3.10
				else: # file
					po.unlink()

		else: # summarize
			result_filepath = args.summarize_file
			
			#data = np.load(result_filepath)
			#accum = data['accum']
			#file_size = data['file_size']

			with open(result_filepath,'rb') as f:
				accum, file_size, libinfo, sysinfo, cmd_str = pickle.load(f)

			print(f'Benchmark results loaded from {result_filepath}\n')
			print('The library and system information where it was run:\n')
			print(pretty_lib_version_info_str(libinfo))
			print(pretty_sys_info_str(sysinfo))
			print(f'This result was generated with command line: {cmd_str}\n')



		# Outputs

		figs = [
			summary_plot_io_time(accum=accum, data_dist=(0,1), wr_rd=0, title='Write speed'),
			summary_plot_io_time(accum=accum, data_dist=(0,1), wr_rd=1, title='Read speed'),
			summary_plot_io_rate(accum=accum),
			summary_plot_file_size(file_size=file_size)
		]

		def fig_to_html_str(fig):
			img = BytesIO()
			fig.savefig(img, format='png', bbox_inches='tight', dpi=72.0)
			img.seek(0)
			s = base64.b64encode(img.getvalue()).decode('utf-8')
			html_str = f'<img src="data:image/png;base64, {s}">'
			return html_str

		#TODO switch to use mpld3 in the future when it becomes capable of showing figures with the same quality
		# html_str='\n'.join([mpld3.fig_to_html(fig, no_extras=True) for fig in figs])

		def html_run_info_str(libinfo,sysinfo,cmd_str):
			s=''
			s+=pretty_lib_version_info_str(libinfo, as_html=True)
			s+='<br>'
			s+=pretty_sys_info_str(sysinfo,as_html=True)
			s+='<br>'
			s+=f'<div class="monospace">This result was generated with command line: <b>{html.escape(cmd_str)}</b></div>\n'
			return s

		html_str='\n'.join([fig_to_html_str(fig) for fig in figs])
		html_str += html_run_info_str(libinfo,sysinfo,cmd_str)

		if not args.no_browser:
			print('Showing results in browser tab...')
			display_html_in_tab(html_str,append_headers=True)
			
		if args.save_html_file is not None:
			root, f = os.path.split(args.save_html_file)
			out_fileprefix = os.path.splitext(f)[0] # remove .html or whatever extension
			out_filepath = os.path.join(root,out_fileprefix+'.html')
			if not args.standalone_html:
				html_str = ''
				for i,fig in enumerate(figs):
					png_filename = f'{out_fileprefix}_{i+1}.png'
					html_str += f'<img src="{png_filename}"/>\n'
					fig.savefig(os.path.join(root,png_filename), dpi=72.0)
				html_str += html_run_info_str(libinfo,sysinfo,cmd_str)

			print(f'Saving report to {out_filepath}.')
			with open(out_filepath,'w',encoding='utf-8') as f:
				f.write(add_html_header(html_str))

		if args.notebook:
			plt.show()

		for fig in figs:
			plt.close(fig) # prevents showing them in jupyter


		print()
		print('Finished.')

	except ValueError as e:
		print('\nCannot proceed due to the following:')
		print(e)

	except Exception as e:
		print('\nUnexpected error. Please report to author:')
		print(e)

		print()
		print('-----TRACEBACK------')
		print(traceback.format_exc()) # for debugging

					