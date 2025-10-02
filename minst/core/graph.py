# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 15:25:06 2019

@author: zhangjuefei
"""


class Graph:
    """
    计算图类
    """

    def __init__(self):
        self.nodes = []  # 计算图内的节点的列表
        self.name_scope = None

    def add_node(self, node):
        """
        添加节点
        """
        self.nodes.append(node)

    def clear_jacobi(self):
        """
        清除图中全部节点的雅可比矩阵
        """
        for node in self.nodes:
            node.clear_jacobi()

    def reset_value(self):
        """
        重置图中全部节点的值
        """
        for node in self.nodes:
            node.reset_value(False)  # 每个节点不递归清除自己的子节点的值

    def node_count(self):
        return len(self.nodes)

    def draw(self, ax=None):
        try:
            import matplotlib
            # 强制使用交互式后端（如Qt5Agg），以支持窗口缩放
            matplotlib.use('Qt5Agg', force=True)
            import networkx as nx
            import matplotlib.pyplot as plt
            from matplotlib.colors import ListedColormap
            import numpy as np
        except:
            raise Exception("Need Module networkx 或未安装 matplotlib Qt5 后端")

        G = nx.Graph()

        already = []
        labels = {}
        for node in self.nodes:
            # 获取节点的值字符串
            value_str = ""
            try:
                if hasattr(node, "value") and node.value is not None:
                    import numpy as np
                    value_arr = np.array(node.value)
                    if value_arr.size == 1:
                        value_str = "v={:.3f}".format(float(value_arr))
                    else:
                        # 判断是否为二维及以上，且行数大于8
                        if value_arr.ndim >= 2 and value_arr.shape[0] > 8:
                            shown = value_arr[:5]
                            value_str = "v=" + np.array2string(shown, precision=3, separator=',', suppress_small=True)
                            value_str += "\n..."
                        else:
                            value_str = "v=" + np.array2string(value_arr, precision=3, separator=',', suppress_small=True)
            except Exception:
                value_str = "v=?"
            # 获取雅可比字符串
            jacobi_str = ""
            try:
                if hasattr(node, "jacobi") and node.jacobi is not None:
                    jacobi_arr = np.array(node.jacobi)
                    if jacobi_arr.size == 1:
                        jacobi_str = "J={:.3f}".format(float(jacobi_arr))
                    else:
                        # 判断是否为二维及以上，且行数大于8
                        if jacobi_arr.ndim >= 2 and jacobi_arr.shape[0] > 8:
                            shown = jacobi_arr[:8]
                            jacobi_str = "J=" + np.array2string(shown, precision=3, separator=',', suppress_small=True)
                            jacobi_str += "\n..."
                        else:
                            jacobi_str = "J=" + np.array2string(jacobi_arr, precision=3, separator=',', suppress_small=True)
            except Exception:
                jacobi_str = "J=?"
            # 组装标签
            labels[node] = (
                node.__class__.__name__ +
                ("({:s})".format(str(node.dim)) if hasattr(node, "dim") else "") +
                ("\n" + value_str if value_str else "") +
                ("\n" + jacobi_str if jacobi_str else "")
            )
            for c in node.children:
                if {node, c} not in already:
                    G.add_edge(node, c)
                    already.append({node, c})
            for p in node.parents:
                if {node, p} not in already:
                    G.add_edge(node, p)
                    already.append({node, c})

        if ax is None:
            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(111)

        ax.clear()
        ax.axis("on")
        ax.grid(True)

        pos = nx.spring_layout(G, seed=42)

        # 有雅克比的变量节点
        cm = plt.cm.Reds
        nodelist = [n for n in self.nodes if n.__class__.__name__ ==
                    "Variable" and n.jacobi is not None]
        colorlist = [np.linalg.norm(n.jacobi) for n in nodelist]
        nx.draw_networkx_nodes(G, pos, nodelist=nodelist, node_color=colorlist, cmap=cm, edgecolors="#666666",
                               node_size=2000, alpha=1.0, ax=ax)

        # 无雅克比的变量节点
        nodelist = [n for n in self.nodes if n.__class__.__name__ ==
                    "Variable" and n.jacobi is None]
        nx.draw_networkx_nodes(G, pos, nodelist=nodelist, node_color="#999999", cmap=cm, edgecolors="#666666",
                               node_size=2000, alpha=1.0, ax=ax)

        # 有雅克比的计算节点
        nodelist = [n for n in self.nodes if n.__class__.__name__ !=
                    "Variable" and n.jacobi is not None]
        colorlist = [np.linalg.norm(n.jacobi) for n in nodelist]
        nx.draw_networkx_nodes(G, pos, nodelist=nodelist, node_color=colorlist, cmap=cm, edgecolors="#666666",
                               node_size=2000, alpha=1.0, ax=ax)
        
        # 无雅克比的中间
        nodelist = [n for n in self.nodes if n.__class__.__name__ !=
                    "Variable" and n.jacobi is None]
        nx.draw_networkx_nodes(G, pos, nodelist=nodelist, node_color="#999999", cmap=cm, edgecolors="#666666",
                               node_size=2000, alpha=1.0, ax=ax)

        # 边
        nx.draw_networkx_edges(G, pos, width=2, edge_color="#014b66", ax=ax)
        nx.draw_networkx_labels(G, pos, labels=labels, font_weight="bold", font_color="#6c6c6c", font_size=12,
                                font_family='arial', ax=ax)
        
        plt.show()
        # 若仍无法缩放，请确保已安装 PyQt5，并在命令行/脚本中运行，而非在部分 IDE/Notebook 环境
        # plt.savefig("computing_graph.png")  # save as png


# 全局默认计算图
default_graph = Graph()
