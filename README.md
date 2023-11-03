# cuda_learning 入门
记录学习Cuda编程的项目

## One Day:

任务清单：配置GPU环境以及创建GPU版的Hello world!

### 一、配置GPU编程环境

主要参考[这篇文章](https://blog.csdn.net/chen565884393/article/details/127905428)

首先检查N卡的驱动，并安装对应的CUDA，然后安装CUDNN

因为我的是GTX1060，所以只能支持CUDA11.4，注意这里我才过坑，安装了11.5,11.6版本在后面对cu程序进行编译的时候会报错，因此要注意显卡能支持的最高CUDA版本，然后根据CUDA版本确定 Visual Studio的版本，

因为官网默认下载最新版本，实际上是不支持古早的CUDA版本的，因此可以根据上面文章提到连接查询官方CUDA支持的VS范围，**注意此处的版本十分重要，如果版本错误会导致后面编译失败**

我最开始下载的是VS 2022 community 的版本，即使在里面选择安装2019,2017的编译器仍然会报错误，提示头文件找不到，或者版本的问题，改回2019版本的就没问题了。

### 二、Hello world

对于GPU编程主要分为两部分，一部分是主机(host)，另一部分则是设备(Device)

主机代码主要在CPU上运行，设备代码则是在GPU上运行完成

我们可以在CPU上运行逻辑分支，然后调用GPU进行计算，但这里GPU的计算与主机是异步的。

调用GPU的函数之后，CPU的控制权依然会往下运行，如果CPU后面的代码可以主动调用同步操作`cudaDeviceSynchronize`函数来同步，这里CPU会阻塞等待GPU完成计算。

同时也可以调用`cudaMemcpy`在主机和设备间拷贝数据时，这里CPU也会阻塞来隐式同步数据。

## Two Day:

GPU编程主要语法与C语言类似，主要是多了一些特殊标识符

$$
\begin{array}{c|c|c|c}
\hline
限定符 & 执行 & 调用 & 备注\\
\_\_{global}\_\_ & 设备端执行 & 主机调用，也可以从算力3以上的设备中调用 & 返回类型必须是void \\
\_\_{device}\_\_ & 设备端执行 & 仅能从设备中调用  \\
\_\_{host}\_\_ & 主机端执行 & 仅能从主机调用 & 可省略 \\ 
\hline
\end{array}
$$

### 代码要点

对于GPU调用时，其线程创建模型为两个部分，第一个部分叫网格(grid),第二个叫线程块(block),线程块包含了N个线程，结构图如下：

```
                                            每一个Grid
----------------------------      -------------------------------
|        |        |        |      |         |         |         |
| grid1  | grid2  | grid3  |      | block1  | block2  | block3  |
|--------------------------|      |-----------------------------|
|        |        |        |      |         |         |         |
|        |        |        | -->  |         |         |         |
|--------------------------|      |-----------------------------|
|        |        |        |      |         |         |         |
|        |        |        |      |         |         |         |
----------------------------      -------------------------------
```
Grid和Block都由一个三维坐标标识(dim3)，每个`(x,y,z)` 唯一确定一个网格或者一个块。

因此我们可以创建一个维度`(3,3,3)`的网格，每个网格又是一个维度为`(4,4,4)`的线程块。因此总线程数为 $3^3(网格数) * 4^3(线程数)$

1. 网格的维度由线程块的数量来表示
2. 线程块的维度由线程数来表示   

在代码`MatrixSum`中：
```
dim3 block = (16);
dim3 grid = ((matrix_len + block.x - 1) / block.x);
```
这里计算所需要网格数的代码`((matrix_len + block.x - 1) / block.x)` 是为了让所有的数据能被网格所包括，因此加上`block.x-1`,这也是为了规避除法的向下取整，导致无法囊括到所有数据。

```
------------------------------------------------------------
| data1 | data2 |       |       |       |       |       |  |
|       |       |       |       |       |       |       |  |
------------------------------------------------------------
                                                            ^
                                                            |
-----------------------------------------------------------------
| block1| block2|       |       |       |       |       |       |
|       |       |       |       |       |       |       |       |
-----------------------------------------------------------------
    1       2       3       4       5       6       7       8
```

# GPU优化

## 3.1GPU架构

GPU有多个流式多处理器(SM)，执行时每一个SM分配多个线程块，SM分配的线程块数由其中的资源来决定

CUDA中采用单指令多线程架构(`SIMT`)，每32个线程为一组称为线程束(`warp`)
线程束中的所有线程同时执行相同的指令

每一个SM都将分配给他的所有线程块的线程按照线程束进行划分，其中一个线程块被分配到一个SM后会一直存在其中，直到完成线程任务

```     
               -------    -------------
           |——>|warp | -->| 32 thread |  --|     ---------
           |   -------    -------------    |---> | block |
-------    |   -------    -------------    |     ---------
| SM  |  --|——>|warp | -->| 32 thread |  --|
-------    |   -------    -------------
           |   -------    -------------
           |——>|warp | -->| 32 thread |
               -------    -------------
```

**warp才是SM上的执行单位**，因此不同线程束之间的进度可能会不一致，也就会导致线程块之间不同的线程以不同的速度前进。

编译指令：
```
nvcc -O3 按照级别3来优化主机代码
nvcc -G -g 生成Debug信息
```

## 3.2线程束

在硬件上线程都是一维排开，尽管线程块可能是1,2,3维。同时根据`threadIdx.x`的连续值来划分线程束，对于多维的线程块，同样可以根据下列公式转换成一维排列。

一维线程 `(x)`：
$$
thread_{i} = threadIdx.x
$$

二维线程 `(x,y)`：
$$
thread_{i} = threadIdx.x + threadIdx.y \times blockDim.x
$$

三维线程块 `(x,y,z)`
$$
thread_{i} = threadIdx.z \times  (blockDim.x \times blockDim.y) + threadIdx.y \times blockDim.x + threadIdx.x
$$

可以看到线程坐标按照 x为内部维度，y作为外部维度，z作为最外面的维度进行递增。

注意到：$一个线程块中的线程束的数量 = 向上取整({一个线程块中的线程数量 \over 线程束大小})$

因此**线程束不会跨线程块进行分配**，如果当一个线程块的线程数不是线程束大小的偶数倍会造成资源浪费。并会影响线程束的调用，从而影响效率

### 3.2.1线程束分化

GPU对于带有逻辑判断的分支语句，例如`if...else...`并没有复杂的分支预测的能力。*但对于同一个线程束中所有线程必须在同一周期中执行相同的指令*，因此分支语句会导致一个线程束中不同的线程走到不同的分支语句，这就是线程束的分化。

**线程束的分化会明显导致性能下降**

对于代码：`chapter3/branch.cu`，关闭编译器的分支预测优化功能进行运行测试。
编译：`nvcc -g -G branch.cu -o branch`

运行：`sudo ./ncu --metrics smsp__sass_average_branch_targets_threads_uniform.pct branch`

可以看到在RTX3050上分支效率结果为：
```
mathKernal2:80%
mathKernal3:100%
mathKernal4:71.43%
```
线程束级别的分支效率几乎没有分化的分支

每个线程束的资源都保存在SM单元中，进行线程束切换的时候不会产生性能损失

同时每个SM内核只有固定数量的共享内存和寄存器组，
因此每个线程消耗的寄存器越少，SM能同时处理的线程束越多
每个线程块消耗的共享内存越多，SM能同时处理的线程块就越少

### 3.2.2 延迟隐藏
指令发出和完成这段时钟周期被称为指令延迟

在指令延迟这段时间中，每个调度器都有一个符合条件的线程束可用于执行，那么就能隐藏这段延迟（流水线）

延迟分为：算术指令延迟/内存指令延迟

隐藏延迟所需要的活跃线程束数量可以由如下公式进行计算：$线程束数量 = 线程束的延迟\times吞吐量$

吞吐量由SM中每个周期的操作数确定，SM中以线程束为执行单位，因此一个周期会同时有线程束大小的操作执行(warpSize)，上述式子换算成操作单位可以表示成：$操作数 = 指令延迟\times吞吐量$

吞吐量的单位为：操作/周期
指令延迟单位为：周期

根据操作数的需要同时很容易反推需要多少个线程，线程束以及每个SM至少多少个线程束。

同时不要忽略SM的线程束数量同时也受限于硬件资源

### 3.2.3 占用率

占用率的计算公式如下：
$$
占用率 = {活跃线程束 \over 最大线程束}
$$

最大线程束数量可以在设备信息中查询，通过`cudaGetDeviceProperties`获取设备参数信息的结构体，`maxThreadsPerMultiProcessor/warpSize` 可以得到最大的线程束数量

例如本机为`NVIDIA GeForce RTX 3050 Laptop GPU`,其中最大的线程束数量为`maxThreadsPerMultiProcessor/warpSize = 1536/32 = 48`

可以对编译器添加参数`--ptxas-options=-v`
或在CMakeLists.txt中添加`set(CUDA_NVCC_FLAGS --ptxas-options=-v)` 来查看每个核函数使用了多少个寄存器以及共享内存的资源。

提高利用率我们需要设置合适的线程块配置。过大过小都会影响资源的利用率。

小线程块：会在所有资源被充分利用之前到达SM的线程束数量的限制

大线程块：会导致每个SM中每个线程可用的硬件资源过少

网格和线程块大小划分准则：

1. 保持每个块中的线程是**线程束大小的倍数**
2. 避免块太小，每个块至少有表128或者256个线程
3. 根据内核资源调整块的大小
4. 块的数量要远多于SM的数量，达到足够的并行

### 3.2.4 同步

线程块之间的同步: `__device void __syncthreads()`
系统级同步：`cudaDevicesSynchronize()`

线程块是以线程束为单位执行，因此同一个线程块中的不同线程束会处于不同的程序点，所以提供`__syncthreads`来同步块中的所有线程

同步之前线程块中的所有线程产生的共享内存或全局内存的修改操作等，在同步后会对线程块中的所有线程可见，并且是安全的。**因此可用于线程间的通讯**



## 3.3 性能指标

老版本cuda可以使用nvprof来查看程序运行的各种指标,现在逐渐替换成ncu工具来检查程序运行的性能指标。

更多参数对照可以参考官方提供的CUDA手册`《NSIGHT COMPUTE COMMAND LINE INTERFACE》` [Ref:](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html#nvprof-guide)


### 3.3.1 SM占用率

```
nvprof --metrics achieved_occupancy ./xxx
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active ./xxx
```
该指标越高越好

核函数运行时间
```
nvprof ./xxx
nsys nvprof ./xxx
```

在矩阵求和的例子中，不同参数下SM占用率和耗时分别为：
```
(32,32):53.24%  22.10ms
(32,16):70.08%  21.60ms
(16,32):73.10%  21.04ms
(16,16):78.04%  21.19ms
```

从上面结果可见第二种比第一种有更多的块，有着更好性能，但第四中比第三种也有更多的块，但性能并没有更好的提升，虽然SM占用率较高

### 3.3.2 内存读取效率

```
./ncu --metrics l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second ./xxx
```

可以看到结果如下：
```
(32,32):78.99 Gb/s 
(32,16):89.97 Gb/s
(16,32):90.65 Gb/s
(16,16):91.21 Gb/s
```
其中第四中具有最高的内存读取效率，但运行时间并没有第三种情况快，所以高内存读取效率并不代表着有最好的性能

### 3.3.3 全局加载效率

```
./ncu --metrics smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct ./xxx
```
可以看到结果如下：
```
(32,32):100%
(32,16):100%
(16,32):100%
(16,16):100%
```

我们可以结合上述三个指标来判断最佳的块组合

## 3.4 归约问题

对于一个数组求和或者求最大值的问题而言，使用CPU顺序求和在数据量比较大的情况下是比较复杂的，时间复杂度`O(n)`,因此可以考虑使用并行归约来进行求解。

并行归约的主要思想是，对于一个线程块划分数据区域，每个数据块处理一个区域，然后逐层将数据结果汇总

参见代码:[reduceInteger.cu:matrixSum](/chapter3/reduceInteger.cu)

方法`matrixSum`主要方式是如图所示：
```
            -------------------------------------
data:       |  0  |  1  |  2  |  3  |  4  |  5  |
            -------------------------------------
               |     |     |     |     |     |
            -------------------------------------
threadId:   |  0  | (1) |  2  | (3) |  4  | (5) |
            -------------------------------------
               |     |     |     |     |     |   
               -------     -------     ------- 
               |           |           |
            -------------------------------------
data:       |  0  |     |  2  |     |  4  |     |
            -------------------------------------
```

其中偶数线程负责进行相邻步长的相加,奇数线程则不进行处理。`matrixSumNeighbored` 方法则是在线程的处理上进行了排序整合

方法`matrixSumNeighbored`主要方式是如图所示：
```
            -------------------------------------
data:       |  0  |  1  |  2  |  3  |  4  |  5  |
            -------------------------------------
               |     |     |     |     |     |
            -------------------------------------
threadId:   |  0  |     |  1  |     |  2  |     |
            -------------------------------------
               |     |     |     |     |     |   
               -------     -------     ------- 
               |           |           |
            -------------------------------------
data:       |  0  |     |  2  |     |  4  |     |
            -------------------------------------
```

使用相邻的线程分别间隔一个步长区处理不同的数据区域进行归并

这两种方法总线程数并没变，只是改变了活跃线程的排布，对于`512`个线程而言，第一种方法按照`32`划分线程束共有`16`个线程束.
每个线程束中有一半的线程处于非活跃状态，而第二种将活跃线程全部集中在前`8`个线程束中,由于线程束中每个线程执行的命令操作要保持一样，因此第一种方法会因为线程的分化影响性能。而第二种则完全避免了这种情况。

**所以确保线程束中减少线程分化有利于提高性能**

同第二种方法类似，`matrixSumInterleaved`使用交叉归约的方式进行求和。主要流程如下：
```
            -------------------------------------
data:       |  0  |  1  |  2  |  3  |  4  |  5  |
            -------------------------------------
               |     |     |     |     |     |
               -------------------     |     |
               |     |     |           |     |
               V     -------------------     |
                     |     |                 |
                     V     -------------------
                           |
                           V
            -------------------------------------
threadId:   |  0  |  1  |  2  |     |     |     |
            -------------------------------------
               |     |     |
            -------------------------------------
data:       |  0  |  1  |  2  |     |     |     |
            -------------------------------------
```

每个线程先按数据长度的一半进行求和处理，然后结果汇集到前半段，然后继续处理前半段数据进行求和。注意边界判断，每次只有小于步长的线程才进行计算。并且同方法二一样活跃线程集中在块的前半部分。

`__syncthreads`块内数据同步，每次加法计算完，确保每块其他线程完成计算。

这三种方法的SM利用率，内存加载速率以及全局加载率如下：
```
matrixSum            96.80%  91.14Gb/s 25.05%
matrixSumNeighbored  93.05% 191.70Gb/s 25.05%
matrixSumInterleaved 91.92%  59.95Gb/s 96.78%
```
从结果可以看到交错合并计算效率最高。

## 3.5 展开归约
展开归约是指在一个循环里将一次执行复制成多次的计算，这样可以将所有归约的次数继续缩短。
展开的次数称为展开因子

对上一节方法三进行更改，每个线程块处理两块数据内容。在进行块内合并之前，先进行块间数据累加，然后进行块内求和。可以看到效率更快。相比较与方法三拥有更高的内存加载速率。

**所以一个方法中有更多的独立内存操作，会导致内存有更高的吞吐量，从而性能提升**

展开归约的吞吐量和展开因子之间成正比。

### 3.5.1 线程内归约展开

对于方法四除了在块间进行归约展开以外，还能在块内归约展开。见方法五，将最后步长为`[8,4,2,1]`的结果展开求和，能使得合并效率更高。

### 3.5.2 完全展开归约
如果已知一个线程内的循环次数，可以手动将其完全展开，以扩大内存的吞吐量，提高执行速率

### 3.5.3 模板
对于一个cuda函数，我们可以使用模板添加一个变量，这个变量会在编译器编译的时候进行编译优化。使用方式如下：
```
template<data_type data_name> __global__ void function(){}
```

编译器在进行分支检查的时候，会自动删除关于`data_name`别且不满足条件的分支，以达到减少分支的目的。

## 3.6 动态并行

即在设备函数运行过程中，根据需要再次调用设备函数进行计算。典型的例子就是嵌套。

嵌套执行分为了父子线程，父线程创建并调用子线程，只有在子线程完成后父线程才会完成。

显示同步：在父线程中设立栅栏点，父线程会等待子线程完成到达栅栏点。
隐式同步：父线程结束时会等待子线程完成。

动态并行的最大嵌套深度为24

## 3.7 总结

1. 线程束是SM的执行调度单位，线程束的大小由设备`warpSize`确定
2. 确保块中的线程数是`warpSize`的整数倍
3. 块中的线程数不能太大也不能太少，至少[128-256]
4. 块的数量不能太少，以遮盖内存延迟
5. 归约问题
6. 展开归约有助于减少循环，增加可并行操作



# 内存模型

## 4.1 内存结构

1. 寄存器：线程私有
2. 本地内存：线程私有，生命周期在于线程
3. 共享内存：线程块拥有，生命周期同线程块
4. 全局内存：全部线程拥有，生命周期同整个程序
5. 常量内存：全部线程拥有（只读），生命周期同整个程序
6. 纹理内存：全部线程拥有（只读），生命周期同整个程序

### 4.1.1 寄存器(片上)

寄存器用于保存线程中自定义的变量。超出硬件上限会使用本地内存替代，并且会影响运行效率

硬件会更具SM上线程块的数量对寄存器进行平分，因此核函数需要的寄存器越少，SM上能够活跃的线程块越多

除了依赖硬件的自行资源划分，我们也可以通过自定义块的数量来改变寄存器的数量。
```
__global__ void __lanch_bounds__(maxThreadsPerBlock,minBlocksPerMultiprocessor)
```
maxThreadsPerBlock：每个块最大的线程数

minBlocksPerMultiprocessor：每个SM中最小的常驻线程块数

同样可以使用编译参数`-maxrregcount=32`来指定所有核函数中寄存器数量的最大值

### 4.1.2 本地内存(片外)

占用空间较大的结构体或者数组会保存在本地内存，以及不满足寄存器保存条件的变量都会保存在本地内存，本地内存本质同全局内存在同一块区域

### 4.1.3 共享内存(片上)

使用`__shared__`标识的变量，存放于共享内存中，共享内存被线程块共享，并且在片上，因此速率更高。共享内存的使用同样会影响SM上活跃的线程束

**共享内存在核函数内声明和定义**

这块区域可以配合`__syncthreads()`作线程块的同步用

共享内存同一级缓存共同使用`64K`的片上内存,可以通过`cudaFuncSetCacheConfig`来指定共享内存和一级缓存的大小。


### 4.1.4 常量内存(片外)

使用`__constant__`标识，常量内存必须定义在核函数外以及全局空间内。大小只有固定的`64K`，常量内存拥有只读属性，因此要从主设备通过`cudaMemcpyToSymbol`往里面写数据.
这个内存区域的读写能力要优于全局内存，适合保存的例如数学系数等

### 4.1.5 纹理内存(片外)

纹理内存拥有硬件支持的滤波和插值

### 4.1.6 全局内存(片外)

全局内存可以被所有的SM访问到，全局内存的创建分为静态创建和动态创建

静态创建：使用`__device__`标识的变量

动态创建：`cudaMalloc()`来创建全局内存

**全局内存的访问通过`32,64,128`字节的内存事务进行访问，并且会内存对齐:这三种事务加载时首地址必须被`32,64,128`整除。**

一般来说用于加载内存请求的事务越多，数据的使用效率越低。

使用`__device__`标识的变量从主机赋值时，通过`cudaMemcpyToSymbol`或者`cudaMemcpyFromSymbol`来写入/读取数据

同样可以通过`cudaGetSymbolAddress`获取全局变量的地址,然后配合`cudaMemcpy`来复制数据到**全局内存变量**中。

## 4.2 内存分配

### 4.2.3 固定内存分配

cuda程序内存分配流程如下：
```
--------            -------------------------------
|主机内存| --copy--> | cuda驱动在内存中创建的固定内存区域|
--------            -------------------------------
                                    |
                                    V
                                --------
                                |设备内存|
                                --------
```

主机进行内存拷贝的同时多了一步创建固定内存区域的步骤，我们可以通过`cudaMallocHost`直接在主机中创建固定内存区域，并通过`cudaFreeHost`来释放，这样效率会比`[cudaMalloc,cudaMemcpy]`一套要高。同时这个区域主机也能直接访问。

注：固定内存分配和释放的成本更高，但是提供了更高的传输吞吐量，当数据量较大的使用固定内存分配效果会更好。

### 4.2.4 零拷贝内存

通常主机和设备间不能直接互相访问变量和数据，零拷贝内存提供了一个特例。

使用零拷贝主要优势有：
1. 设备内存不足可以利用主机内存
2. 避免主机和设备间的显示内存传输

零拷贝也是属于固定内存，通过`cudaHostAlloc`创建一块主机内存并映射到设备内存区域，同时使用`cudaFreeHost`来释放.

在设备端通过`cudaHostGetDevicePointer`来获取这块映射的内存指针。

注：设备端每次映射的传输与访问都会经过PCIe总线，会导致效率过低，这也是我们需要避免过多的主从设备见数据传输的原因。

因此共享少量数据的时候使用零拷贝内存是个不错的选择，数据量越大传输延迟越高。

### 4.2.5 统一虚拟寻址

对于计算能力2.0以上的设备支持统一虚拟寻址(UVA)，设备**不需要**通过`cudaHostGetDevicePointer`来手动获取主机上的内存映射，直接使用`cudaHostAlloc`分配的主机指针即可，可以交由设备自行进行寻址。

### 4.2.6 统一内存寻址

在统一内存寻址中，提供了一块内存池，底层系统在统一内存空间自动在主机和设备之间进行数据传输。

通过`__managed__`在文件范围和全局范围内添加一个静态托管变量，或者通过主机端调用`cudaMallocManaged`创建一块动态分配的托管内存


## 4.3 内存访问

### 4.3.1 对齐与合并访问
GPU内存请求特性：核函数的内存请求，通常是在物理内存上以128字节或者32字节内存事务来实现。

一级缓存一行是128字节，映射到内存中就是128字节的对齐段。

二级缓存则是32字节的内存事务进行访问

可以通过编译器选择是否开启一级缓存

禁止一级缓存编译参数：`-Xptxas -dlcm=cg`

开启一级缓存编译参数：`-Xptxas -dlcm=ca`

核函数对数据的访问产生一个内存事务，如果内存事务的首地址是对一级缓存大小或者二级缓存大小的偶数倍时，就出现了访问对齐，同一个线程束中的线程访问的数据是一个连续的内存块就会出现合并内存访问。

**合并和对齐会最大化全局内存吞吐量**

全局内存的加载顺序，如果一级缓存没被禁用，那么全局内存首先由一级缓存(128字节)尝试加载，如果被禁用则用二级缓存(32字节)尝试加载，二级缓存也被禁用那么会直接交由`DRAM`(128字节)的内存事务实现。

注意到，一级缓存的加载是128字节为单位，对于一个线程束而言，每个线程请求一个4字节变量，并且分布均匀不重复的内存空间，一次加载事务刚好满足，此时利用率为`100%`,但如果需要的数据不关于128字节对齐，那么需要两个加载事务，利用率为${128 \over 128 \times 2} = 50\%$，这个时候使用粒度更细的32字节加载,可以看到加载率为${128 \over 32 \times 5} = 80\%$，所以开启一级缓存在有的情况下并一定能提高总线加载效率

**二级缓存的粒度是同内存粒度的**


### 4.3.3 全局内存写入

数据写入内存前都会通过二级缓存进行操作，也就是以32字节的粒度进行写入。
内存事务可以分为一段，两段，四段事务执行。分别对应`32,64,128`字节

### 4.3.4 结构体数组(AoS)和数组结构体(SoA)

两种数据布局分别如下所示：
```
AoS:                    SoA:           
---------------------   ---------------------
|x|y|x|y|x|y|...    |   |x|x|x|y|y|y|...    |
---------------------   --------------------- 
```

AoS的每次加载都会加载一半的`x`一半的`y`值,对于只用某一项的核函数会损失一般的加载效率。

对于单一命令多数据的架构，使用SoA更合适。例如：线程束中每个线程处理的指令相同，所以使用的数据项也是相同的，利于提高数据利用率的方式就是SoA模式。

### 4.3.5 性能调整

1. 对齐和合并数据加载
2. 足够多的线程块，掩盖内存延迟
3. **展开技术**

其中展开技术提高IO并行操作带来的效率提高最为显著，并且同时能减少内存读写事务。

## 4.4 矩阵转置

矩阵转置分为两种:按行读取转置成列，按列读取转置成行。

### 4.4.1 按行转置

```

    --------col----------         --------row----------
    |             ny    |         |     |             |
    |              |    |         |     |             |
    ------nx-------N-----         |     nx            |
    |              |    |         |     |             |
 row|              |    |      col|     |             |
    |              |    |         |     |             |
    |              |    |         --ny--N--------------
    ---------------------         |     |             |
                                  ---------------------
```
在这里`nx`作为横轴(行方向)的变化量，对于`N(nx,ny)` 在转置矩阵中的关系为，`N(nx,ny) --> N(ny,nx)`转换成一维矩阵的表达式：$ny \times col + nx \rightarrowtail nx \times row + ny$

### 4.4.1 按列转置

```

    --------col----------         --------row----------
    |             nx    |         |     |             |
    |              |    |         |     |             |
    ------ny-------N-----         |     ny            |
    |              |    |         |     |             |
 row|              |    |      col|     |             |
    |              |    |         |     |             |
    |              |    |         --nx--N--------------
    ---------------------         |     |             |
                                  ---------------------
```
在这里`nx`作为竖轴(列方向)的变化量，对于`N(ny,nx)` 在转置矩阵中的关系为，`N(ny,nx) --> N(nx,ny)`转换成一维矩阵的表达式：$nx \times col + ny \rightarrowtail ny \times row + nx$



单独测试数据读取和复制`(copyRow,copyCol)`有如下结果：
```
        读取效率, 写入效率, 读取带宽,  写入带宽
copyCol: 12.5%   12.5%   74.23Gb/s  74.28Gb/s
copyRow:  100%    100%   41.68Gb/s  41.74Gb/s
```
对于按列转置和按行转置`(transposeNaiveRow,transposeNaiveCol)`可以发现有如下结果：
```
    读取效率, 写入效率,     读取带宽,               写入带宽
Col: 12.5%   100%   194.82Gb/s(交叉读取)  24.35Gb/s(合并写入)
Row:  100%  12.5%     9.04Gb/s(合并读取)  72.32Gb/s(交叉写入)
```

可以看到按列转置的读取带宽最高，这是因为在一级缓存的作用下，每个线程加载128字节的数据，但是只使用其中一个，但是其相邻线程进行操作时会有更高的缓存命中率。

同时使用按行拷贝会有更短的执行时间。按行拷贝时，读取效率和写入效率都会更高，这是因为读取和写入都通过一个事务即可。

对于按列转置而言，相比较与按行具有很高读取带宽相差20倍左右，而写入带宽相差仅仅3倍。这导致按列转置具有更低的执行时间。

当使用展开优化后，展开因子为4,`(transposeNaiveRow_unroll4,transposeNaiveCol_unroll4)`可以发现有如下结果：

```
            读取效率, 写入效率,     读取带宽,               写入带宽
Col_unroll4: 12.5%   100%      281.99Gb/s(交叉读取)  35.25Gb/s(合并写入)
Row_unroll4: 100%    12.5%     10.28Gb/s(合并读取)    82.23Gb/s(交叉写入)
```

按行展开在内存带宽的吞吐量上有些微提升，这也`transposeNaiveRow_unroll4`执行时间要比`transposeNaiveRow`少的原因。因为有更高的可并发内存操作。

对于按列展开同样相比较与`transposeNaiveCol`拥有更高的读取带宽和写入带宽，同样是因为可并行的内存读写操作导致。因此拥有更快的执行效率

**对于矩阵操作优先使用按列操作**

## 4.4.3 对角坐标系

另一种优化方式可以对网格的坐标系进行优化,传统我们一般使用直角坐标系如下图所示

```
        直角坐标系(y,x)                     对角坐标系(y,x)
-------------------------        -------------------------
|(0,0)|(0,1)|(0,2)|(0,3)|        |(0,0)|(0,1)|(0,2)|(0,3)|
-------------------------        -------------------------
|(1,0)|(1,1)|(1,2)|(1,3)|        |(1,3)|(1,0)|(1,1)|(1,2)|
-------------------------        -------------------------
|(2,0)|(2,1)|(2,2)|(2,3)|        |(2,2)|(2,3)|(2,0)|(2,1)|
-------------------------        -------------------------
|(3,0)|(3,1)|(3,2)|(3,3)|        |(3,1)|(3,2)|(3,3)|(3,0)|
-------------------------        ------------------------- 
```
直角坐标系和对角坐标系的转换公式：

$$
y_对 = y_直 \\
x_对 = (x_直 - y_直 + gridDim.x) \% gridDim.x
$$

对角坐标系转换成直角坐标系
$$
y_直 = y_对 \\
x_直 = (x_对 + y_对) \% gridDim.x
$$

内存的加载最终由DRAM硬件完成，在直角坐标系中，将线程快映射到内存区域时，由于排列过于紧密，很容易出现分区冲突(计算机组成原理存储部分)，而使用对角坐标系能一定程度的拉开相邻线程快加载数据的距离。从而减少分区冲突现象。

但是对角坐标系带来的提升不如按列读取


## 4.4.5 瘦块

**使用更瘦的块能提高存储的效率**

在`transpose`函数中 将块的大小改为`<<<8,32>>>`,加载效率和写入效率相比较与`<<<16,32>>>`提升了一倍

## 4.4.6 统一内存管理

使用统一内存管理同样能提升处理效率

# 5 共享内存和常量内存

## 5.1 共享内存

共享内存是片上内存，全局内存是版载内存，共享内存提供了更高的访问速率并有更低的延迟。

### 5.1.1 共享内存的分配

在核函数内或者文件范围内，通过`__shared__`来标识变量。表示该变量是共享内存。或者申明一个共享内存的多维数组。

如果对于数组长度未知，我们可以通过`extern`来申明，然后通过在每个核函数调用时通过第三个参数来定义大小。**这个只支持一维数组**。

```
extern __shared__ int temp[];

function<<<grid,block,byteSize>>>();
```

每个线程块创建时都会创建一定的共享内存，线程块中的所有线程共享该块内存区域，并且具有相同的生命周期，如果多个线程同时访问共享内存的统一变量，该变量会通过多播发送给其他线程。

共享内存等分为32个同样大小的内存模型，叫做**存储体**，这些存储体可以同时被访问，这个数量同`warpSize`,如果线程束中的每个线程访问不同的存储体，那么可以用一个内存事务完成。否则就是多个事务，这样会降低加载效率

### 5.1.3 存储体冲突

多个地址请求落在同一块内存区域时，就会发生存储体冲突。

存储体请求分为三种情况：
1. 并行访问：多个地址请求多个存储体
2. 串行访问：多个地址请求一个存储体
3. 广播访问：单个地址请求一个存储体

### <div id="5.1.3.1">5.1.3.1 访问模式</div>
对于共享内存的访问模式来说，不同的计算平台有不同的存储体宽度`(32位,64位)`
也就是每个存储体的一个读写操作的单位。并且**连续的32位字映射到连续的存储体中**

因此32位的访问平台有如下的地址到存储体的计算映射公式：

$$idx = {{address \over 4} \% 32}$$

访问地址除以4个字节，转换成4字节索引，然后对32个存储体取余

```
      ------------------------------------   ------
   0x | 00 | 04 | 08 | 0C | 10 | 14 | 18 |...| 80 |
      ------------------------------------   ------
         |    |    |    |    |    |    |       
      ------------------------------------   ------
   /4 | 00 | 01 | 02 | 03 | 04 | 05 | 06 |...| 32 |
      ------------------------------------   ------
         |    |    |    |    |    |    |       
      ------------------------------------   ------
  %32 | 00 | 01 | 02 | 03 | 04 | 05 | 06 |...| 00 |
      ------------------------------------   ------
```
**邻近的字被分到不同的存储体中**，所以同一线程束中的线程**访问邻近的地址时，不会有存储体冲突**。

对于64位的访问平台，地址到存储体的映射公式为：

$$idx = {{address \over 8} \% 32}$$

访问地址除以8个字节，转换成8字节索引，然后对32个存储体取余

注意两个线程访问64中的子字时(每个线程只需要其中的32位数据)，并不会产生存储体冲突，因为只需要一个64字的加载事务。

#### 5.1.3.4 内存填充

内存填充是解决存储体冲突的方法之一，通过在列后面添加一列元素可以使得原来数据排序错乱开。在访问同一列元素时可以落到不同的存储体中。主要能够解决按列写存储体冲突的问题。

```
-----------------          -----------------
| 0 | 1 | 2 | 3 |          | 0 | 1 | 2 | 3 |
-----------------    -->   -----------------
| 4 | 5 | 6 | 7 |          | x | 4 | 5 | 6 |
-----------------          -----------------
```
注：填充会增加每个线程块使用共享内存的大小，从而影响每个SM中可活跃的线程束数量。

共享内存通过4字节还是8字节访问可通过:`cudaDeviceSetSharedMemConfig`来进行配置

### 5.1.4 栅栏和块同步

在前面我们介绍了`__syncthreads()`用来使块内线程全部同步到指定点

这里介绍一种全局内存同步的方法：内存栅栏，这个方法可以确保栅栏前内存的所有操作，在栅栏后对所有线程是可见的。

内存栅栏分为三个粒度：
1. 线程块级别：`__threadfence_block()`
2. 网格级别：`__threadfence();`
3. 系统级别:`__threadfence_system()`

线程块级别的可以保证线程块对**共享内存和全局内存**所有的内存操作 在该线程块内是可见的。

网格级别的，会挂起调用的线程，直到在网格级别内保证**全局内存**的读写在栅栏后是可见的

系统级别则是可以决定主机和设备之间的数据同步，主要针对全局内存，主机的固定内页内存，设备中的内存。

### 5.1.5 volatile 修饰符

该修饰符可以避免编译器优化：将数据暂存于寄存器或者本地内存中。这个修饰符修饰的变量会实时更新到全局内存中。

## 5.2 共享内存布局

更具[5.1.3.1](#5.1.3.1)中 共享内存的访问模式我们可以看到，访问相邻数据时，是不会产生存储体冲突，因此对于一个二维数组 `__shared__ int temp[w][h];`而言，其是按照行优先进行存储成一维数组，因此我们使用**行优先访问**会避免存储体冲突。使用列会产生一半的冲突

更具线程的`threadIdx.x`连续，推荐使用 `temp[threadIdx.y][threadIdx.x]`

**注：按列写入会产生许多存储体冲突，可以通过添加列数来解决**