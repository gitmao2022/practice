'''
@Description  : file content
@Version      : 1.0
@Author       : gitmao2022
@Date         : 2025-03-23 20:46:05
@LastEditors  : gitmao2022
@LastEditTime : 2026-09-09 22:07:57
@FilePath     : operate_node.py
@Copyright (C) 2025  by ${gitmao2022}. All rights reserved.
'''


import numpy as np

from .node import Node


def fill_diagonal(to_be_filled, filler):
    """
    将 filler 矩阵填充在 to_be_filled 的对角线上
    """
    assert to_be_filled.shape[0] / \
        filler.shape[0] == to_be_filled.shape[1] / filler.shape[1]
    n = int(to_be_filled.shape[0] / filler.shape[0])

    r, c = filler.shape
    for i in range(n):
        to_be_filled[i * r:(i + 1) * r, i * c:(i + 1) * c] = filler

    return to_be_filled




class Add(Node):
    """
    加法运算
    """

    def compute_value(self):
        assert len(self.parents) == 2
        value=self.parents[0].value + self.parents[1].value
        return value

    def get_jacobi(self, parent):
        if parent.value.shape == self.value.shape:
            # Not use broadcast
            return np.eye(self.dimension())
        else:
            # Broadcast case for y = x + b, where x is (N, M) and b is (1, M).
            # d vec(y) / d vec(b) = kron(ones(N, 1), I_M), shape (N*M, M).
            if len(self.value.shape) == 2 and len(parent.value.shape) == 2:
                n, m = self.value.shape
                if parent.value.shape == (1, m):
                    return np.kron(np.ones((n, 1)), np.eye(m))

            raise ValueError(
                f"Unsupported broadcast in Add.get_jacobi: self.shape={self.value.shape}, parent.shape={parent.value.shape}"
            )


class MatMul(Node):
    """
    矩阵乘法
    """

    def compute_value(self):
        assert len(self.parents) == 2 and self.parents[0].shape[
            1] == self.parents[1].shape[0]
        return np.dot(self.parents[0].value,self.parents[1].value)

    def get_jacobi(self, parent):
        """
        将矩阵乘法视作映射，求映射对参与计算的矩阵的雅克比矩阵。
        """

        # 很神秘，靠注释说不明白了
        zeros = np.zeros((self.dimension(), parent.dimension()))
        if parent is self.parents[0]:
            return fill_diagonal(zeros, self.parents[1].value.T)
        else:
            jacobi = fill_diagonal(zeros, self.parents[0].value)
            row_sort = np.arange(self.dimension()).reshape(
                self.shape[::-1]).T.ravel()
            col_sort = np.arange(parent.dimension()).reshape(
                parent.shape[::-1]).T.ravel()
            return jacobi[row_sort, :][:, col_sort]



class Reshape(Node):
    """
    改变父节点的值（矩阵）的形状
    """

    def __init__(self, *parent, **kargs):
        Node.__init__(self, *parent, **kargs)

        self.to_shape = kargs.get('shape')
        assert isinstance(self.to_shape, tuple) and len(self.to_shape) == 2


    def compute_value(self):
        self.value = self.parents[0].value.reshape(self.to_shape)

    def get_jacobi(self, parent):
        assert parent is self.parents[0]
        return np.mat(np.eye(self.dimension()))


class Multiply(Node):
    """
    两个父节点的值是相同形状的矩阵，将它们对应位置的值相乘
    """

    def compute_value(self):
        return np.multiply(self.parents[0].value, self.parents[1].value)

    def get_jacobi(self, parent):
        if parent is self.parents[0]:
            return np.diag(self.parents[1].value.flatten())
        else:
            return np.diag(self.parents[0].value.flatten())


class Convolve(Node):
    """
    使用一个二维卷积核对批量图像数据进行离散卷积（same 卷积，输出图像尺寸与输入一致）。

    输入数据恒为二维结构：第 1 维是图像数量 N，第 2 维是图像拉平后的数据（H*W）。
    本节点在前向与反向传播的计算内部，将每行拉平数据还原为 (H, W) 的二维图像；
    一旦跳出本节点的计算（compute_value / get_jacobi 返回），
    数据依然是 (N, H*W) 的二维原始结构，以便后续运算。

    父节点：
        parents[0]: 图像数据，形状 (N, H*W)。
        parents[1]: 卷积核，形状 (KH, KW)。

    kargs:
        image_shape: 二元组 (H, W)，图像真实的高和宽，必须提供。
    """

    def __init__(self, *parents, **kargs):
        assert len(parents) == 2
        Node.__init__(self, *parents, **kargs)
        self.image_shape = kargs.get('image_shape')
        assert self.image_shape is not None, \
            "image_shape (height, width) is required"
        self.image_shape = tuple(self.image_shape)
        assert len(self.image_shape) == 2, \
            "image_shape should be a tuple (height, width)"

    def _validated_values(self):
        data = self.parents[0].value
        kernel = self.parents[1].value
        height, width = self.image_shape
        assert data.ndim == 2, "data should have shape (N, H*W)"
        assert data.shape[1] == height * width, \
            "flattened dimension of data should equal H*W of image_shape"
        assert kernel.ndim == 2, "kernel should have shape (KH, KW)"
        return data, kernel

    def _to_images(self, data):
        """
        将 (N, H*W) 的批量拉平数据还原为 (N, H, W) 的二维图像序列。
        """
        height, width = self.image_shape
        return data.reshape(-1, height, width)

    def _convolve_single(self, image, kernel):
        """
        使用一个二维卷积核对一张二维图像进行离散卷积。
        """
        height, width = image.shape
        kernel_height, kernel_width = kernel.shape
        half_height, half_width = kernel_height // 2, kernel_width // 2
        result = np.zeros((height, width))

        for row in range(height):

            # 卷积核滑动时每次计算的开始和结束行
            row_start = max(0, row - half_height)
            row_stop = min(height, row + kernel_height - half_height)
            # 用 row_start - (row - half_height) 更易理解：
            # kernel_start 本来应该从第 0 行开始，因为原图像截去了
            # row_start - (row - half_height) 行，所以 kernel_start 相应地向下移动
            kernel_row_start = row_start - row + half_height

            for col in range(width):
                col_start = max(0, col - half_width)
                col_stop = min(width, col + kernel_width - half_width)
                kernel_col_start = col_start - col + half_width

                window = image[row_start:row_stop, col_start:col_stop]
                kernel_window = kernel[
                    kernel_row_start:kernel_row_start + row_stop - row_start,
                    kernel_col_start:kernel_col_start + col_stop - col_start
                ]
                result[row, col] = np.sum(window * kernel_window)

        return result

    def compute_value(self):
        data, kernel = self._validated_values()
        images = self._to_images(data)
        result = np.array([self._convolve_single(image, kernel)
                           for image in images])
        # 将 (N, H, W) 的卷积结果拉平，还原成 (N, H*W) 的二维原始结构
        return result.reshape(data.shape)

    def _jacobi_single_data(self, image_shape, kernel):
        """
        单张图像的卷积结果对输入图像的雅可比矩阵。

        数学背景：
            前向公式（same 卷积，核不翻转）：
                result[r, c] = Σ_i Σ_j  image[r-hh+i, c-hw+j] * kernel[i, j]
            其中 hh = KH//2, hw = KW//2，求和只对不越界的图像位置进行。

            由上式可知，输出像素 result[r, c] 对某个输入像素 image[r', c']
            的偏导数，就是卷积核上与其配对的那个权重：
                ∂result[r, c] / ∂image[r', c'] = kernel[r'-r+hh, c'-c+hw]
            （若 (r', c') 不在以 (r, c) 为锚点的核覆盖范围内，则偏导为 0）

        矩阵布局：
            输出图像和输入图像都按“行优先”拉平成一维向量：
                拉平下标 = 行号 * width + 列号
            雅可比矩阵形状为 (H*W, H*W)：
                行 = 输出像素的拉平下标，列 = 输入像素的拉平下标。
            即 jacobi[输出像素, 输入像素] = 该输出对该输入的偏导数。

        参数：
            image_shape: (H, W)，图像的真实高宽（只需形状，无需具体数值，
                         因为偏导数只取决于卷积核，不取决于图像内容）。
            kernel: (KH, KW) 的卷积核。
        """
        height, width = image_shape
        kernel_height, kernel_width = kernel.shape
        half_height, half_width = kernel_height // 2, kernel_width // 2
        image_dim = height * width
        jacobi = np.zeros((image_dim, image_dim))

        # 遍历每一个输出像素 (row, col)
        for row in range(height):
            for col in range(width):
                # 该输出像素在拉平向量中的下标（雅可比的行号）
                output_index = row * width + col

                # 遍历卷积核的每个位置 (kernel_row, kernel_col)，
                # 找出它当前覆盖的是图像的哪个像素
                for kernel_row in range(kernel_height):
                    # 核第 kernel_row 行对应的图像行：
                    # 核中心（第 half_height 行）对准锚点 row，
                    # 所以核第 i 行对应图像第 row + i - half_height 行
                    input_row = row + kernel_row - half_height
                    if not 0 <= input_row < height:
                        continue  # 越界位置相当于乘 0，偏导为 0，无需写入
                    for kernel_col in range(kernel_width):
                        # 同理，核第 kernel_col 列对应图像的列
                        input_col = col + kernel_col - half_width
                        if 0 <= input_col < width:
                            # 被覆盖的输入像素在拉平向量中的下标（雅可比的列号）
                            input_index = input_row * width + input_col
                            # 前向中 image[input_row, input_col] 乘的正是
                            # kernel[kernel_row, kernel_col]，
                            # 所以偏导数就是这个核权重
                            jacobi[output_index, input_index] = kernel[
                                kernel_row, kernel_col
                            ]

        return jacobi

    def _jacobi_single_kernel(self, image, kernel):
        """
        单张图像的卷积结果对卷积核的雅可比矩阵。

        数学背景：
            由前向公式 result[r, c] = Σ_i Σ_j image[r+i-hh, c+j-hw] * kernel[i, j]
            可知，输出像素对某个核权重的偏导数，就是与它配对的那个图像像素：
                ∂result[r, c] / ∂kernel[i, j] = image[r+i-hh, c+j-hw]
            （若配对位置越出图像边界，则偏导为 0）

        矩阵布局：
            输出图像按行优先拉平成长度 H*W 的向量，
            卷积核也按行优先拉平成长度 KH*KW 的向量：
                核的拉平下标 = 核行号 * kernel_width + 核列号
            雅可比矩阵形状为 (H*W, KH*KW)：
                行 = 输出像素的拉平下标，列 = 核权重的拉平下标。

        参数：
            image: (H, W) 的单张图像（这里需要具体数值，因为偏导数
                   本身就是图像像素的值）。
            kernel: (KH, KW) 的卷积核（只用其形状确定遍历范围）。
        """
        height, width = image.shape
        kernel_height, kernel_width = kernel.shape
        half_height, half_width = kernel_height // 2, kernel_width // 2
        jacobi = np.zeros((height * width, kernel_height * kernel_width))

        # 遍历每一个输出像素 (row, col)
        for row in range(height):
            for col in range(width):
                # 该输出像素在拉平向量中的下标（雅可比的行号）
                output_index = row * width + col

                # 遍历卷积核的每个位置 (kernel_row, kernel_col)
                for kernel_row in range(kernel_height):
                    # 核第 kernel_row 行当前覆盖的图像行
                    input_row = row + kernel_row - half_height
                    if not 0 <= input_row < height:
                        continue  # 越界：该核权重本次未被使用，偏导为 0
                    for kernel_col in range(kernel_width):
                        # 核第 kernel_col 列当前覆盖的图像列
                        input_col = col + kernel_col - half_width
                        if 0 <= input_col < width:
                            # 前向中 kernel[kernel_row, kernel_col] 乘的正是
                            # image[input_row, input_col]，
                            # 所以偏导数就是这个图像像素的值；
                            # 列号 = 核权重在拉平核向量中的下标
                            jacobi[
                                output_index,
                                kernel_row * kernel_width + kernel_col
                            ] = image[input_row, input_col]

        return jacobi

    def get_jacobi(self, parent):
        data, kernel = self._validated_values()
        assert parent in self.parents

        images = self._to_images(data)

        if parent is self.parents[0]:
            # 每张输出图像只依赖对应的输入图像，
            # 整体雅可比是由单张图像雅可比构成的块对角矩阵
            block = self._jacobi_single_data(images[0].shape, kernel)
            return np.kron(np.eye(len(images)), block)

        # 对卷积核的雅可比：各图像对应的块纵向堆叠
        return np.vstack([self._jacobi_single_kernel(image, kernel)
                          for image in images])


class MaxPooling(Node):
    """
    最大值池化。

    输入数据恒为二维结构：第 1 维是图像数量 N，第 2 维是图像拉平后的数据（H*W）。
    本节点在计算内部将每行拉平数据还原为 (H, W) 的二维图像，
    逐张做最大池化压缩后，再把结果拉平还原为 (N, out_H*out_W) 的二维结构。

    父节点：
        parents[0]: 图像数据，形状 (N, H*W)。

    kargs:
        image_shape: 二元组 (H, W)，图像真实的高和宽，必须提供。
        size: 二元组 (KH, KW)，池化窗口的高和宽。
        stride: 二元组 (SH, SW)，行 / 列方向的步长。

    输出形状 (N, out_H*out_W)，其中：
        out_H = ceil(H / SH),  out_W = ceil(W / SW)
    """

    def __init__(self, *parent, **kargs):
        Node.__init__(self, *parent, **kargs)

        # 图像真实高宽
        self.image_shape = kargs.get('image_shape')
        assert self.image_shape is not None, \
            "image_shape (height, width) is required"
        self.image_shape = tuple(self.image_shape)
        assert len(self.image_shape) == 2, \
            "image_shape should be a tuple (height, width)"

        # 池化步长
        self.stride = kargs.get('stride')
        assert self.stride is not None, "stride is required"
        self.stride = tuple(self.stride)
        assert len(self.stride) == 2, "stride should be a tuple (stride_h, stride_w)"

        # 池化窗口尺寸
        self.size = kargs.get('size')
        assert self.size is not None, "size is required"
        self.size = tuple(self.size)
        assert len(self.size) == 2, "size should be a tuple (kernel_h, kernel_w)"

        # 每张图像的最大值位置标记（0/1 矩阵，行 = 输出像素，列 = 输入像素）
        # 批量时为列表，逐图像存储
        self.flag = None

    def _pool_single(self, image):
        """
        对一张 (H, W) 的二维图像做最大池化，返回：
            result: (out_H, out_W) 的池化结果
            flag:   (out_H*out_W, H*W) 的 0/1 矩阵，标记每个输出取自输入的哪个位置
        """
        height, width = image.shape
        image_dim = height * width
        sh, sw = self.stride
        kh, kw = self.size
        hkh, hkw = kh // 2, kw // 2  # 池化窗口高宽的一半

        result = []
        flag = []

        # 以步长滑动窗口（锚点为窗口中心）
        for i in range(0, height, sh):
            row = []
            for j in range(0, width, sw):
                # 窗口边界，越界则裁剪
                top, bottom = max(0, i - hkh), min(height, i + kh - hkh)
                left, right = max(0, j - hkw), min(width, j + kw - hkw)
                window = image[top:bottom, left:right]
                row.append(np.max(window))

                # 记录最大值在原图像中的位置（拉平下标）
                pos = np.argmax(window)
                win_width = right - left
                offset_row = top + pos // win_width
                offset_col = left + pos % win_width
                offset = offset_row * width + offset_col
                tmp = np.zeros(image_dim)
                tmp[offset] = 1
                flag.append(tmp)

            result.append(row)

        # flag 形状 (out_H*out_W, H*W)，正好是本图像池化操作的雅可比矩阵
        return np.array(result), np.array(flag)

    def compute_value(self):
        data = self.parents[0].value
        height, width = self.image_shape
        assert data.ndim == 2 and data.shape[1] == height * width, \
            "data should have shape (N, H*W) with H*W == image_shape[0]*image_shape[1]"

        # 还原成 (N, H, W) 逐张池化
        images = data.reshape(-1, height, width)
        results = []
        flags = []
        for image in images:
            result, flag = self._pool_single(image)
            results.append(result)
            flags.append(flag)

        self.flag = flags
        # 拉平还原为 (N, out_H*out_W) 的二维结构
        return np.array([r.flatten() for r in results])

    def get_jacobi(self, parent):
        assert parent is self.parents[0] and self.flag is not None

        # 单张图像：直接返回该图像的 0/1 标记矩阵
        if len(self.flag) == 1:
            return self.flag[0]

        # 批量：每张输出图像只依赖对应的输入图像，
        # 整体雅可比是由各图像 flag 构成的块对角矩阵。
        block_rows, block_cols = self.flag[0].shape
        jacobi = np.zeros((len(self.flag) * block_rows,
                           len(self.flag) * block_cols))
        for index, block in enumerate(self.flag):
            row_start = index * block_rows
            col_start = index * block_cols
            jacobi[row_start:row_start + block_rows,
                   col_start:col_start + block_cols] = block
        return jacobi


class Concat(Node):
    """
    将多个父节点的值连接成向量
    """

    def compute_value(self):
        assert len(self.parents) > 0

        # 将所有父节点矩阵展平并连接成一个向量
        self.value = np.concatenate(
            [p.value.flatten() for p in self.parents],
            axis=1
        ).T

    def get_jacobi(self, parent):
        assert parent in self.parents

        dimensions = [p.dimension() for p in self.parents]  # 各个父节点的元素数量
        pos = self.parents.index(parent)  # 当前是第几个父节点
        dimension = parent.dimension()  # 当前父节点的元素数量

        assert dimension == dimensions[pos]

        jacobi = np.mat(np.zeros((self.dimension(), dimension)))
        start_row = int(np.sum(dimensions[:pos]))
        jacobi[start_row:start_row + dimension,
               0:dimension] = np.eye(dimension)

        return jacobi


class ScalarMultiply(Node):
    """
    用标量（1x1矩阵）数乘一个矩阵
    """

    def compute_value(self):
        assert self.parents[0].shape() == (1, 1)  # 第一个父节点是标量
        self.value = np.multiply(self.parents[0].value, self.parents[1].value)

    def get_jacobi(self, parent):

        assert parent in self.parents

        if parent is self.parents[0]:
            return self.parents[1].value.flatten().T
        else:
            return np.mat(np.eye(self.parents[1].dimension())) * self.parents[0].value[0, 0]


class Step(Node):

    def compute_value(self):
        self.value = np.where(self.parents[0].value >= 0.0, 1.0, 0.0)

    def get_jacobi(self, parent):
        return np.diag(np.where(self.parents[0].value.flatten() >= 0.0, 0.0, -1.0))


class Welding(Node):

    def compute_value(self):

        assert len(self.parents) == 1 and self.parents[0] is not None
        self.value = self.parents[0].value

    def get_jacobi(self, parent):

        assert parent is self.parents[0]
        return np.mat(np.eye(self.dimension()))

    def weld(self, node):
        """
        将本节点焊接到输入节点上
        """

        # 首先与之前的父节点断开

        if len(self.parents) == 1 and self.parents[0] is not None:
            self.parents[0].children.remove(self)

        self.parents.clear()

        # 与输入节点焊接
        self.parents.append(node)
        node.children.append(self)
