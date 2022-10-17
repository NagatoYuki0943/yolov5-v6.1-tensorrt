# 运行yolov5导出的tensorrt

# 通过yolov5 `export.py` 导出

>https://github.com/ultralytics/yolov5/issues/251

# 通过 `trtexec.exe` 导出

> 先通过yolov5导出onnx，再通过 `trtexec.exe` 导出 engine

```powershell
.\trtexec.exe --help
&&&& RUNNING TensorRT.trtexec [TensorRT v8403] # D:\ai\TensorRT\bin\trtexec.exe --help
=== Model Options ===
  --uff=<file>                UFF model
  --onnx=<file>               ONNX model
  --model=<file>              Caffe model (default = no model, random weights used)
  --deploy=<file>             Caffe prototxt file
  --output=<name>[,<name>]*   Output names (it can be specified multiple times); at least one output is required for UFF and Caffe
  --uffInput=<name>,X,Y,Z     Input blob name and its dimensions (X,Y,Z=C,H,W), it can be specified multiple times; at least one is required for UFF models
  --uffNHWC                   Set if inputs are in the NHWC layout instead of NCHW (use X,Y,Z=H,W,C order in --uffInput)

=== Build Options ===
  --maxBatch                  Set max batch size and build an implicit batch engine (default = same size as --batch)
                              This option should not be used when the input model is ONNX or when dynamic shapes are provided.
  --minShapes=spec            Build with dynamic shapes using a profile with the min shapes provided
  --optShapes=spec            Build with dynamic shapes using a profile with the opt shapes provided
  --maxShapes=spec            Build with dynamic shapes using a profile with the max shapes provided
  --minShapesCalib=spec       Calibrate with dynamic shapes using a profile with the min shapes provided
  --optShapesCalib=spec       Calibrate with dynamic shapes using a profile with the opt shapes provided
  --maxShapesCalib=spec       Calibrate with dynamic shapes using a profile with the max shapes provided
                              Note: All three of min, opt and max shapes must be supplied.
                                    However, if only opt shapes is supplied then it will be expanded so
                                    that min shapes and max shapes are set to the same values as opt shapes.
                                    Input names can be wrapped with escaped single quotes (ex: \'Input:0\').
                              Example input shapes spec: input0:1x3x256x256,input1:1x3x128x128
                              Each input shape is supplied as a key-value pair where key is the input name and
                              value is the dimensions (including the batch dimension) to be used for that input.
                              Each key-value pair has the key and value separated using a colon (:).
                              Multiple input shapes can be provided via comma-separated key-value pairs.
  --inputIOFormats=spec       Type and format of each of the input tensors (default = all inputs in fp32:chw)
                              See --outputIOFormats help for the grammar of type and format list.
                              Note: If this option is specified, please set comma-separated types and formats for all
                                    inputs following the same order as network inputs ID (even if only one input
                                    needs specifying IO format) or set the type and format once for broadcasting.
  --outputIOFormats=spec      Type and format of each of the output tensors (default = all outputs in fp32:chw)
                              Note: If this option is specified, please set comma-separated types and formats for all
                                    outputs following the same order as network outputs ID (even if only one output
                                    needs specifying IO format) or set the type and format once for broadcasting.
                              IO Formats: spec  ::= IOfmt[","spec]
                                          IOfmt ::= type:fmt
                                          type  ::= "fp32"|"fp16"|"int32"|"int8"
                                          fmt   ::= ("chw"|"chw2"|"chw4"|"hwc8"|"chw16"|"chw32"|"dhwc8"|
                                                     "cdhw32"|"hwc"|"dla_linear"|"dla_hwc4")["+"fmt]
  --workspace=N               Set workspace size in MiB.
  --memPoolSize=poolspec      Specify the size constraints of the designated memory pool(s) in MiB.
                              Note: Also accepts decimal sizes, e.g. 0.25MiB. Will be rounded down to the nearest integer bytes.
                              Pool constraint: poolspec ::= poolfmt[","poolspec]
                                               poolfmt ::= pool:sizeInMiB
                                               pool ::= "workspace"|"dlaSRAM"|"dlaLocalDRAM"|"dlaGlobalDRAM"
  --profilingVerbosity=mode   Specify profiling verbosity. mode ::= layer_names_only|detailed|none (default = layer_names_only)
  --minTiming=M               Set the minimum number of iterations used in kernel selection (default = 1)
  --avgTiming=M               Set the number of times averaged in each iteration for kernel selection (default = 8)
  --refit                     Mark the engine as refittable. This will allow the inspection of refittable layers
                              and weights within the engine.
  --sparsity=spec             Control sparsity (default = disabled).
                              Sparsity: spec ::= "disable", "enable", "force"
                              Note: Description about each of these options is as below
                                    disable = do not enable sparse tactics in the builder (this is the default)
                                    enable  = enable sparse tactics in the builder (but these tactics will only be
                                              considered if the weights have the right sparsity pattern)
                                    force   = enable sparse tactics in the builder and force-overwrite the weights to have
                                              a sparsity pattern (even if you loaded a model yourself)
  --noTF32                    Disable tf32 precision (default is to enable tf32, in addition to fp32)
  --fp16                      Enable fp16 precision, in addition to fp32 (default = disabled)
  --int8                      Enable int8 precision, in addition to fp32 (default = disabled)
  --best                      Enable all precisions to achieve the best performance (default = disabled)
  --directIO                  Avoid reformatting at network boundaries. (default = disabled)
  --precisionConstraints=spec Control precision constraint setting. (default = none)
                                  Precision Constaints: spec ::= "none" | "obey" | "prefer"
                                  none = no constraints
                                  prefer = meet precision constraints set by --layerPrecisions/--layerOutputTypes if possible
                                  obey = meet precision constraints set by --layerPrecisions/--layerOutputTypes or fail
                                         otherwise
  --layerPrecisions=spec      Control per-layer precision constraints. Effective only when precisionConstraints is set to
                              "obey" or "prefer". (default = none)
                              The specs are read left-to-right, and later ones override earlier ones. "*" can be used as a
                              layerName to specify the default precision for all the unspecified layers.
                              Per-layer precision spec ::= layerPrecision[","spec]
                                                  layerPrecision ::= layerName":"precision
                                                  precision ::= "fp32"|"fp16"|"int32"|"int8"
  --layerOutputTypes=spec     Control per-layer output type constraints. Effective only when precisionConstraints is set to
                              "obey" or "prefer". (default = none)
                              The specs are read left-to-right, and later ones override earlier ones. "*" can be used as a
                              layerName to specify the default precision for all the unspecified layers. If a layer has more than
                              one output, then multiple types separated by "+" can be provided for this layer.
                              Per-layer output type spec ::= layerOutputTypes[","spec]
                                                    layerOutputTypes ::= layerName":"type
                                                    type ::= "fp32"|"fp16"|"int32"|"int8"["+"type]
  --calib=<file>              Read INT8 calibration cache file
  --safe                      Enable build safety certified engine
  --consistency               Perform consistency checking on safety certified engine
  --restricted                Enable safety scope checking with kSAFETY_SCOPE build flag
  --saveEngine=<file>         Save the serialized engine
  --loadEngine=<file>         Load a serialized engine
  --tacticSources=tactics     Specify the tactics to be used by adding (+) or removing (-) tactics from the default
                              tactic sources (default = all available tactics).
                              Note: Currently only cuDNN, cuBLAS, cuBLAS-LT, and edge mask convolutions are listed as optional
                                    tactics.
                              Tactic Sources: tactics ::= [","tactic]
                                              tactic  ::= (+|-)lib
                                              lib     ::= "CUBLAS"|"CUBLAS_LT"|"CUDNN"|"EDGE_MASK_CONVOLUTIONS"
                              For example, to disable cudnn and enable cublas: --tacticSources=-CUDNN,+CUBLAS
  --noBuilderCache            Disable timing cache in builder (default is to enable timing cache)
  --timingCacheFile=<file>    Save/load the serialized global timing cache

=== Inference Options ===
  --batch=N                   Set batch size for implicit batch engines (default = 1)
                              This option should not be used when the engine is built from an ONNX model or when dynamic
                              shapes are provided when the engine is built.
  --shapes=spec               Set input shapes for dynamic shapes inference inputs.
                              Note: Input names can be wrapped with escaped single quotes (ex: \'Input:0\').
                              Example input shapes spec: input0:1x3x256x256, input1:1x3x128x128
                              Each input shape is supplied as a key-value pair where key is the input name and
                              value is the dimensions (including the batch dimension) to be used for that input.
                              Each key-value pair has the key and value separated using a colon (:).
                              Multiple input shapes can be provided via comma-separated key-value pairs.
  --loadInputs=spec           Load input values from files (default = generate random inputs). Input names can be wrapped with single quotes (ex: 'Input:0')
                              Input values spec ::= Ival[","spec]
                                           Ival ::= name":"file
  --iterations=N              Run at least N inference iterations (default = 10)
  --warmUp=N                  Run for N milliseconds to warmup before measuring performance (default = 200)
  --duration=N                Run performance measurements for at least N seconds wallclock time (default = 3)
  --sleepTime=N               Delay inference start with a gap of N milliseconds between launch and compute (default = 0)
  --idleTime=N                Sleep N milliseconds between two continuous iterations(default = 0)
  --streams=N                 Instantiate N engines to use concurrently (default = 1)
  --exposeDMA                 Serialize DMA transfers to and from device (default = disabled).
  --noDataTransfers           Disable DMA transfers to and from device (default = enabled).
  --useManagedMemory          Use managed memory instead of separate host and device allocations (default = disabled).
  --useSpinWait               Actively synchronize on GPU events. This option may decrease synchronization time but increase CPU usage and power (default = disabled)
  --threads                   Enable multithreading to drive engines with independent threads or speed up refitting (default = disabled)
  --useCudaGraph              Use CUDA graph to capture engine execution and then launch inference (default = disabled).
                              This flag may be ignored if the graph capture fails.
  --timeDeserialize           Time the amount of time it takes to deserialize the network and exit.
  --timeRefit                 Time the amount of time it takes to refit the engine before inference.
  --separateProfileRun        Do not attach the profiler in the benchmark run; if profiling is enabled, a second profile run will be executed (default = disabled)
  --buildOnly                 Exit after the engine has been built and skip inference perf measurement (default = disabled)

=== Build and Inference Batch Options ===
                              When using implicit batch, the max batch size of the engine, if not given,
                              is set to the inference batch size;
                              when using explicit batch, if shapes are specified only for inference, they
                              will be used also as min/opt/max in the build profile; if shapes are
                              specified only for the build, the opt shapes will be used also for inference;
                              if both are specified, they must be compatible; and if explicit batch is
                              enabled but neither is specified, the model must provide complete static
                              dimensions, including batch size, for all inputs
                              Using ONNX models automatically forces explicit batch.

=== Reporting Options ===
  --verbose                   Use verbose logging (default = false)
  --avgRuns=N                 Report performance measurements averaged over N consecutive iterations (default = 10)
  --percentile=P              Report performance for the P percentage (0<=P<=100, 0 representing max perf, and 100 representing min perf; (default = 99%)
  --dumpRefit                 Print the refittable layers and weights from a refittable engine
  --dumpOutput                Print the output tensor(s) of the last inference iteration (default = disabled)
  --dumpProfile               Print profile information per layer (default = disabled)
  --dumpLayerInfo             Print layer information of the engine to console (default = disabled)
  --exportTimes=<file>        Write the timing results in a json file (default = disabled)
  --exportOutput=<file>       Write the output tensors to a json file (default = disabled)
  --exportProfile=<file>      Write the profile information per layer in a json file (default = disabled)
  --exportLayerInfo=<file>    Write the layer information of the engine in a json file (default = disabled)

=== System Options ===
  --device=N                  Select cuda device N (default = 0)
  --useDLACore=N              Select DLA core N for layers that support DLA (default = none)
  --allowGPUFallback          When DLA is enabled, allow GPU fallback for unsupported layers (default = disabled)
  --plugins                   Plugin library (.so) to load (can be specified multiple times)

=== Help ===
  --help, -h                  Print this message
```

> 导出

```powershell
.\trtexec.exe --onnx=yolov5s.onnx --saveEngine=yolov5s.engine --fp16	# Enable fp16 precision, in addition to fp32
.\trtexec.exe --onnx=yolov5s.onnx --saveEngine=yolov5s.engine --int8	# Enable int8 precision, in addition to fp32
.\trtexec.exe --onnx=yolov5s.onnx --saveEngine=yolov5s.engine --best	# Enable all precisions to achieve the best performance
```


# 参考

https://github.com/dacquaviva/yolov5-openvino-cpp-python

https://mmdeploy.readthedocs.io/zh_CN/latest/tutorial/06_introduction_to_tensorrt.html