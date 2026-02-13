'''
@Description  : file content
@Version      : 1.0
@Author       : gitmao2022
@Date         : 2025-03-30 16:22:30
@LastEditors  : gitmao2022
@LastEditTime : 2025-12-07 20:20:46
@FilePath     : graph.py
@Copyright (C) 2025. All rights reserved.
'''
# -*- coding: utf-8 -*-


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

    def clear_changeable_value(self):
        """
        清除图中所有可变节点的值
        """
        for node in self.nodes:
            #判断node的类型为Variable且node.trainable为False时跳过清除value操作
            if node.__class__.__name__=='Variable' :
                pass
            else:
                node.clear_value()
    def clear_jacobi(self):
        """
        清除图中全部节点的雅可比矩阵
        """
        for node in self.nodes:
            node.clear_jacobi()

    def node_count(self):
        return len(self.nodes)

    def draw(self, ax=None):
        try:
            import matplotlib
            # 强制使用交互式后端（如Qt5Agg），以支持窗口缩放
            matplotlib.use('Qt5Agg', force=True)
            import networkx as nx
            import matplotlib.pyplot as plt
            import numpy as np
        except Exception:
            raise Exception("Need Module networkx 或未安装 matplotlib Qt5 后端")

        G = nx.Graph()

        # build graph and labels
        already = []
        labels_full = {}
        labels_short = {}
        for node in self.nodes:
            # value string
            value_str = ""
            try:
                if hasattr(node, "value") and node.value is not None:
                    value_arr = np.array(node.value)
                    if value_arr.size == 1:
                        value_str = "v={:.3f}".format(float(value_arr))
                    else:
                        # show longer preview for larger arrays: take first 20 rows when large
                        if value_arr.ndim >= 2 and value_arr.shape[0] > 50:
                            shown = value_arr[:20]
                            value_str = "v=" + np.array2string(shown, precision=3, separator=',', suppress_small=True)
                            value_str += "\n..."
                        else:
                            value_str = "v=" + np.array2string(value_arr, precision=3, separator=',', suppress_small=True)
            except Exception:
                value_str = "v=?"
            jacobi_str = ""
            try:
                if hasattr(node, "jacobi") and node.jacobi is not None:
                    jacobi_arr = np.array(node.jacobi)
                    if jacobi_arr.size == 1:
                        jacobi_str = "J={:.3f}".format(float(jacobi_arr))
                    else:
                        # show longer preview for larger jacobi arrays
                        if jacobi_arr.ndim >= 2 and jacobi_arr.shape[0] > 50:
                            shown = jacobi_arr[:20]
                            jacobi_str = "J=" + np.array2string(shown, precision=3, separator=',', suppress_small=True)
                            jacobi_str += "\n..."
                        else:
                            jacobi_str = "J=" + np.array2string(jacobi_arr, precision=3, separator=',', suppress_small=True)
            except Exception:
                jacobi_str = "J=?"
            # full label contains detailed info (for tooltip)
            labels_full[node] = (
                node.__class__.__name__ +
                ("({:s})".format(str(node.dim)) if hasattr(node, "dim") else "") +
                ("\n" + value_str if value_str else "")
            )
            # short label shown on plot: only name and size
            try:
                if hasattr(node, 'dim'):
                    size_str = str(getattr(node, 'dim'))
                else:
                    import numpy as _np
                    vtmp = getattr(node, 'value', None)
                    if vtmp is None:
                        size_str = 'N'
                    else:
                        size_str = str(_np.array(vtmp).shape)
            except Exception:
                size_str = 'N'
            labels_short[node] = node.__class__.__name__ + ("(" + size_str + ")")
            for c in getattr(node, 'children', []):
                if {node, c} not in already:
                    G.add_edge(node, c)
                    already.append({node, c})
            for p in getattr(node, 'parents', []):
                if {node, p} not in already:
                    G.add_edge(node, p)
                    already.append({node, p})

        if ax is None:
            # 尝试查找并复用之前由本 draw 创建的图形/坐标轴，避免重复打开新窗口。
            fig = None
            ax = None
            try:
                for num in plt.get_fignums():
                    ftemp = plt.figure(num)
                    reuse_ax = None
                    # 优先选择带有 _graph_refresh 的 axis（上次绘图留下的标记）
                    for a in ftemp.axes:
                        if hasattr(a, '_graph_refresh'):
                            reuse_ax = a
                            break
                    if reuse_ax is not None:
                        fig = ftemp
                        ax = reuse_ax
                        break
                    # 其次检查 figure 是否有旧的 _graph_cid 标记，作为回退
                    if hasattr(ftemp, '_graph_cid') and ftemp.axes:
                        fig = ftemp
                        ax = ftemp.axes[0]
                        break
            except Exception:
                fig = None
                ax = None

            if fig is None or ax is None:
                fig = plt.figure(figsize=(12, 12))
                ax = fig.add_subplot(111)
            else:
                # 若复用已有 figure，先断开之前的回调，避免重复触发或引用已失效的闭包变量
                try:
                    if hasattr(fig, '_graph_cid'):
                        try:
                            fig.canvas.mpl_disconnect(fig._graph_cid)
                        except Exception:
                            pass
                except Exception:
                    pass

        ax.clear()
        ax.axis("on")
        ax.grid(True)

        pos = nx.spring_layout(G, seed=42)

        # prepare node lists and colors
        all_nodes = list(pos.keys())
        xs = np.array([pos[n][0] for n in all_nodes])
        ys = np.array([pos[n][1] for n in all_nodes])

        # color by jacobi norm when available
        color_vals = []
        sizes = []
        for n in all_nodes:
            try:
                if hasattr(n, 'jacobi') and n.jacobi is not None:
                    color_vals.append(np.linalg.norm(n.jacobi))
                else:
                    color_vals.append(0.0)
            except Exception:
                color_vals.append(0.0)
            # size by dimension if present
            try:
                sizes.append(2000 if not hasattr(n, 'dim') else 500 + 50 * int(getattr(n, 'dim', 1)))
            except Exception:
                sizes.append(1000)

        cmap = plt.cm.Reds
        sc = ax.scatter(xs, ys, c=color_vals, cmap=cmap, s=sizes, edgecolors="#666666", picker=True)

        # draw edges
        for u, v in G.edges():
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            ax.plot([x1, x2], [y1, y2], color="#014b66", linewidth=2)

        # draw short labels (name and size only)
        txt_artists = {}
        for n in all_nodes:
            lbl = labels_short.get(n, n.__class__.__name__)
            t = ax.text(pos[n][0], pos[n][1], lbl, fontsize=9, ha='center', va='center', zorder=3)
            txt_artists[n] = t

        # interactive tooltip annotation
        annot = ax.annotate("", xy=(0,0), xytext=(20,20), textcoords="offset points",
                            bbox=dict(boxstyle="round", fc="w"), arrowprops=dict(arrowstyle="->"))
        annot.set_visible(False)

        def update_annot(ind):
            # ind is dict returned by contains
            idx = ind["ind"][0]
            node = all_nodes[idx]
            annot.xy = (xs[idx], ys[idx])
            # show full detailed label on hover
            text = labels_full.get(node, txt_artists[node].get_text())
            annot.set_text(text)
            annot.get_bbox_patch().set_facecolor('#ffffdd')
            annot.get_bbox_patch().set_alpha(0.9)

        def hover(event):
            vis = annot.get_visible()
            if event.inaxes == ax:
                cont, ind = sc.contains(event)
                if cont:
                    update_annot(ind)
                    annot.set_visible(True)
                    plt.draw()
                else:
                    if vis:
                        annot.set_visible(False)
                        plt.draw()

        fig = ax.figure
        # keep connection id on figure so linter doesn't mark it unused and to allow disconnect later
        fig._graph_cid = fig.canvas.mpl_connect("motion_notify_event", hover)

        # add refresh function to update colors/labels in-place
        def refresh():
            # recompute colors and label texts
            new_colors = []
            for n in all_nodes:
                try:
                    if hasattr(n, 'jacobi') and n.jacobi is not None:
                        new_colors.append(np.linalg.norm(n.jacobi))
                    else:
                        new_colors.append(0.0)
                except Exception:
                    new_colors.append(0.0)
                # update stored labels: full (detailed) and short (on-plot)
                try:
                    # recompute detailed label
                    value_str = ""
                    if hasattr(n, "value") and n.value is not None:
                        v = np.array(n.value)
                        if v.size == 1:
                            value_str = "v={:.3f}".format(float(v))
                        else:
                            value_str = "v=" + np.array2string(v, precision=3, separator=',', suppress_small=True)
                    labels_full[n] = n.__class__.__name__ + ("({:s})".format(str(n.dim)) if hasattr(n, "dim") else "")
                    if value_str:
                        labels_full[n] += "\n" + value_str
                    # short label remains name(size)
                    try:
                        if hasattr(n, 'dim'):
                            size_str = str(getattr(n, 'dim'))
                        else:
                            vtmp = getattr(n, 'value', None)
                            if vtmp is None:
                                size_str = 'N'
                            else:
                                size_str = str(np.array(vtmp).shape)
                    except Exception:
                        size_str = 'N'
                    labels_short[n] = n.__class__.__name__ + ("(" + size_str + ")")
                    txt_artists[n].set_text(labels_short[n])
                except Exception:
                    pass
            sc.set_array(np.array(new_colors))
            fig.canvas.draw_idle()

        # expose refresh on ax for external callers
        ax._graph_refresh = refresh

        # 使用非阻塞显示，保证函数返回后程序继续执行；若后端不支持再降级为阻塞显示
        try:
            plt.ion()
            plt.show(block=False)
            # 短暂停顿以确保 GUI 事件循环得到处理，窗口正确刷新
            plt.pause(0.001)
        except Exception:
            # 回退到阻塞显示，确保在极端环境下仍能工作
            plt.show()


# 全局默认计算图
default_graph = Graph()
