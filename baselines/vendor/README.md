# 基线兼容依赖

本目录仅包含 Netzob 运行时需要的 `bintrees 2.2.0` 纯 Python 实现。
源码取自 PyPI 发布包 `bintrees-2.2.0.zip`，SHA-256 为
`e180658d90789855dcb0e7d1eb2bfebc452d60c5b48e74de16b502d61a8352d1`。
该依赖采用 MIT License，许可证保存在 `BINTREES_LICENSE.txt`。

未修改 AVLTree 算法；仅省略了可选的 Cython 加速文件，使服务器无需联网编译即可
加载与原包等价的纯 Python 回退实现。

NEMESYS 的非交互内存评测还需要两个纯 Python 依赖，均直接从 PyPI wheel
解包且未修改：

- `bitstring 3.1.9`，SHA-256
  `e3e340e58900a948787a05e8c08772f1ccbe133f6f41fe3f0fa19a18a22bbf4f`；
- `kneed 0.8.5`，SHA-256
  `2f3fbd4e9bd808e65052841448702c41ea64d5fc78735cbfc97ab25f08bd9815`。
- `tabulate 0.9.0`，SHA-256
  `024ca478df22e9340661486f85298cff5f6dcdba14f3813e8830015b9ed1948f`。

两者的原始许可证和包元数据保留在各自的 `.dist-info` 目录中。
