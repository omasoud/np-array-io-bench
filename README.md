# np-array-io-bench
Tool for benchmarking the saving and loading speed of numpy arrays

<br/><br/>
### Usage
```
usage: npabench.py [-h] [-s FILE | --max-size MAX_SIZE] [--no-browser]
                   [--save-html-file FILE] [--standalone-html] [--notebook]

Benchmark load/save speeds and summarize results graphically. A results file
is generated and saved. The tool can be also run just to read a results file
and show the results graphically using the --summarize-file argument.

optional arguments:
  -h, --help            show this help message and exit
  -s FILE, --summarize-file FILE
                        Only generate summary graphs based on provided results
                        file. This will not run the benchmark.
  --max-size MAX_SIZE   Maximum file size (must be a power of 2) to benchmark
                        (e.g, 8MB, 4GB, 128KB). Default: 1MB. Caution: large
                        sizes can take a very long time or run out of memory
                        or disk space. 16GB takes about 90 minutes on a fast
                        computer.
  --no-browser          Do not launch a browser tab to display the results.
  --save-html-file FILE
                        If desired, provide filename so that html report gets
                        saved to it.
  --standalone-html     By default the html will reference generated png files
                        for the figres. If desired, endocded. But if desired
                        this option can encode the pngs directly in the html
                        (making it larger but standalone).
  --notebook            Plot results in Jupyter Notebook.
```

<br/><br/>

## Benchmark Result:

https://omasoud.github.io/npabench/npabench_result.html


Copyright (c) 2022 O. Masoud