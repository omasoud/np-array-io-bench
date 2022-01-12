import os
import gc
import numpy as np
import pickle
import h5py
import tables
import zarr
from timeit import default_timer as timer
import datetime as dt
import tempfile
from pathlib import Path
import argparse
import traceback
import matplotlib.pyplot as plt
import matplotlib
import cpuinfo

def elapsed():
	try:
		return timer() - elapsed.last_timestamp
	except:
		return None
	finally:
		elapsed.last_timestamp = timer()


def profile_write(fmt,arr):
	gc.disable()
	elapsed()
	format_rw[fmt][1](arr)
	e = elapsed()
	gc.enable()
	return e

def profile_read(fmt):
	gc.disable()
	elapsed()
	format_rw[fmt][2]()
	e = elapsed()
	gc.enable()
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
		return f["data"][()]	
def pytables_w(fpath,a):
	with tables.open_file(fpath, 'w') as f:
		gcolumns = f.create_group(f.root, 'columns', 'data')
		f.create_array(gcolumns, 'data', a, 'data')
def pytables_r(fpath):
	with tables.open_file(fpath, 'r') as f:
		return f.root.columns.data[()]

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


def summary_plot(accum, data_dist, wr_rd, title):
	data_dist = data_dist if isinstance(data_dist, tuple) else (data_dist,)
	max_pwr = accum.shape[1]-1
	y=np.nanmean(accum,axis=-1)[data_dist,:,:,wr_rd].mean(axis=0) # shape (max_pwr,8)
	y_no_outliers = np.ma.array(y,mask=get_outlier_mask(np.log(y),axis=1,thresh=9.0)) # outliers decided in the log domain
	y_min = y_no_outliers.min(axis=1)
	y_max = y_no_outliers.max(axis=1)
	x=np.broadcast_to(np.array([2**k for k in range(max_pwr+1)])[:,np.newaxis],y.shape)



	fig=plt.figure(figsize = (12, 8))
	#plt.gca().xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
	fig.subplots_adjust(
		top=0.95,
		bottom=0.05,
		left=0.05,
		right=0.95,
		hspace=0.2,
		wspace=0.25
	)
	fig.suptitle(title, fontsize=10)

	ax1=plt.subplot(211)

	ax1.loglog(x,y)
	relabel_size_axis(ax1, max_pwr=max_pwr)
	relabel_time_axis(ax1)
	ax1.set_xlim([1,2**max_pwr])
	ax1.set_xlabel('Size',loc='right')
	ax1.set_ylabel('Sec',loc='top')
	ax1.legend(list(format_rw.keys()))

	a,b=13,18
	if max_pwr<b:
		plt.show()
		return
	ax2 = plt.subplot(234)
	ax2.loglog(x,y)
	relabel_size_axis(ax2, max_pwr=max_pwr)
	relabel_time_axis(ax2, yres=.15)
	# 8KB = 	1024*8 = 	2**13
	# 256KB = 	1024*256 = 	2**18
	ax2.set_xlim([2**a,2**b])
	ax2.set_ylim([min(y_min[a],y_min[b]), max(y_max[a],y_max[b])])
	ax2.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
	ax2.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())


	a,b=22,24
	if max_pwr<b:
		plt.show()
		return
	ax3 = plt.subplot(235)
	ax3.loglog(x,y)
	relabel_size_axis(ax3, max_pwr=max_pwr)
	relabel_time_axis(ax3,yres=.25)
	# 4MB	22
	# 16	24
	ax3.set_xlim([2**a,2**b])
	ax3.set_ylim([min(y_min[a],y_min[b]), max(y_max[a],y_max[b])])
	ax3.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
	ax3.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())


	a,b=31,32
	# if max_pwr<b:
	# 	plt.show()
	# 	return
	if max_pwr<=24:
		plt.show()
		return
	elif max_pwr<b:
		a,b=max_pwr-1,max_pwr

	ax4 = plt.subplot(236)
	ax4.loglog(x,y)
	relabel_size_axis(ax4, max_pwr=max_pwr)
	relabel_time_axis(ax4, yres=.1)
	# 2GB	31
	# 4GB	32	
	ax4.set_xlim([2**a,2**b])
	ax4.set_ylim([min(y_min[a],y_min[b]), max(y_max[a],y_max[b])])
	ax4.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
	ax4.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())

	plt.show()

def summary_plot_io_rate(accum):
	data_dist=(0,1)
	max_pwr = accum.shape[1]-1
	y1=np.nanmean(accum,axis=-1)[data_dist,:,:,0].mean(axis=0) # write time, all arrays
	y2=np.nanmean(accum,axis=-1)[data_dist,:,:,1].mean(axis=0) # read time, all arrays

	fig=plt.figure(figsize = (12, 8))
	#plt.gca().xaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
	fig.subplots_adjust(
		top=0.90,
		bottom=0.05,
		left=0.05,
		right=0.95,
		hspace=0.2,
		wspace=0.25
	)
	x=np.broadcast_to(np.array([2**k for k in range(max_pwr+1)])[:,np.newaxis],y1.shape) # y1 and y2 have the same shape
	start_pwr=13


	ax1=plt.subplot(211)
	ax1.loglog(x,x/y1/1024/1024/1024)
	ax1.set_yscale('linear')
	relabel_size_axis(ax1, max_pwr=max_pwr, start_pwr=start_pwr)
	ax1.set_xlim([2**start_pwr,2**max_pwr])
	ax1.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
	ax1.grid(axis='y')
	ax1.set_ylabel('GB/Sec',loc='top')
	ax1.set_title(r'write rate')
	ax1.legend(list(format_rw.keys()))


	ax2=plt.subplot(212)
	ax2.loglog(x,x/y2/1024/1024/1024)
	ax2.set_yscale('linear')
	relabel_size_axis(ax2, max_pwr=max_pwr, start_pwr=start_pwr)
	ax2.set_xlim([2**start_pwr,2**max_pwr])
	ax2.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
	ax2.set_yticks(np.arange(0,5.5,.5))
	ax2.grid(axis='y')
	ax2.set_xlabel('Size',loc='right')
	ax2.set_title(r'read rate')
	fig.suptitle('IO Rate', fontsize=14)

	plt.show()	

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
	ax1.set_xlim([1,2**max_pwr])
	ax1.grid(axis='y')
	ax1.set_ylabel('Compression Ratio',loc='top')
	ax1.legend(list(format_rw.keys()))
	ax1.set_title('random numbers')

	ax2=plt.subplot(212)
	ax2.loglog(x,x/y2)
	ax2.set_yscale('linear')
	relabel_size_axis(ax2, max_pwr)
	ax2.set_xlim([1,2**max_pwr])
	ax2.grid(axis='y')
	ax2.set_xlabel('Size',loc='right')
	ax2.set_title(r'80% sparse random numbers')

	fig.suptitle('File Size', fontsize=14)

	plt.show()	

def get_lib_version_info():
	s='Library versions:\n'
	s+=f'numpy:\t{np.version.version}\n'
	s+=f'pickle:\t{pickle.format_version}\n'
	s+=f'h5py:\t{h5py.version.version}\n'
	s+=f'tables:\t{tables.get_pytables_version()} (using hdf5 version: {tables.hdf5_version})\n'
	s+=f'zarr:\t{zarr.__version__}\n'
	return s

def get_sys_info():
	s='System information:\n'
	s+=cpuinfo.get_cpu_info()['brand_raw']
	return s

if __name__ == '__main__':

	print('Numpy Array File I/O Benchmark. By O. Masoud')

	parser = argparse.ArgumentParser(description='Benchmark load/save speeds and summarize results graphically.')
	group = parser.add_mutually_exclusive_group(required=True)
	group.add_argument('--benchmark', action='store_true', help='Run the benchmark and store the results to a results file.')
	group.add_argument('--summarize', action='store', help='Generate summary graphs based on provided data file.')
	parser.add_argument('--max-power', action='store', default=None, type=int, # Won't use defailt here due to error checking below
					help='Maximum file size (as a power of 2) to benchmark (e.g, 32 means 4GB). Default=10 (1MB). ' 
					'Caution: large sizes can take a very long time or run out of memory or disk space. 34 (16GB) '
					'takes about 90 minutes on a fast computer.')

	args = parser.parse_args()

	if args.summarize and args.max_power is not None:
		raise ValueError('--max-power can be specified only with --benchmark.')

	try:
		if args.benchmark:
			time_str = dt.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
			result_filepath = os.path.abspath(f'numpy_array_bench_{time_str}.pkl')
			print(f'Results will be written to: {result_filepath}')
			print()
			print(get_lib_version_info())
			print()
			print(get_sys_info())

			#MAX_PWR=32 # 32 for 4 gig; 34 for 16 gig
			#MAX_PWR=34
			if args.max_power is None:
				MAX_PWR=10
			else:
				MAX_PWR=args.max_power

			MAX_SIZE = 2**MAX_PWR

			print('Preparing arrays...')
			dtype = np.float32
			uni=np.random.default_rng().random(MAX_SIZE//dtype().itemsize, dtype=dtype)
			arrays={
				'uniform': uni,
				'sparse': uni*(uni<.2)*4 # 80% sparse, rescale to [0,1)
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

			print('Running benchmark...')
			accum = np.full((len(arrays), MAX_PWR+1, len(format_rw), 2, calc_reps(0)),np.nan) # 2 for write&read, 10 for max reps
			file_size = np.full((len(arrays), MAX_PWR+1, len(format_rw)),np.nan)
			for ddist,arr in enumerate(arrays.values()):
				for size_pwr in range(MAX_PWR+1): # 1 byte to 4GB
					total_reps = calc_reps(size_pwr)	
					print(ddist,size_pwr,total_reps)
					for j,fmt in enumerate(format_rw.keys()):
						fpath=get_fpath(fmt) # reuse the same file path for each format
#						for rep in range(min(max(2**(31-size_pwr),4),10)): # some reasonable reps
						for rep in range(total_reps): # some reasonable reps
							accum[ddist,size_pwr,j,0,rep] = profile_write(fmt,arr[:2**size_pwr//dtype().itemsize])
							file_size[ddist,size_pwr,j] = file_or_dir_size(fpath)
							accum[ddist,size_pwr,j,1,rep] = profile_read(fmt)

			#np.savez(result_filepath, accum=accum, file_size=file_size)
			with open(result_filepath,'wb') as f:
				pickle.dump([accum,
							file_size,
							get_lib_version_info(),
							get_sys_info()],f)


			#TODO delete temps 

		else: # summarize
			result_filepath = args.summarize
			
			#data = np.load(result_filepath)
			#accum = data['accum']
			#file_size = data['file_size']

			with open(result_filepath,'rb') as f:
				accum, file_size, lib_version_str, sys_info_str = pickle.load(f)

			print('Benchmark results loaded. Here is some relevant information when it was run:')
			print()
			print(lib_version_str)
			print()
			print(sys_info_str)
			print()

			summary_plot(accum=accum, data_dist=(0,1), wr_rd=0, title='Write speed')
			summary_plot(accum=accum, data_dist=(0,1), wr_rd=1, title='Read speed')
			summary_plot_io_rate(accum=accum)
			summary_plot_file_size(file_size=file_size)

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

					