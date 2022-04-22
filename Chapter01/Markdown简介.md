## Markdown
Markdown是一种轻量化的标记语言。主要用于书写技术文档。

### 参考资料
[Markdown菜鸟教程](https://www.runoob.com/markdown/md-tutorial.html)

### 标题
以`#`起始的行为一级标题，以`##` 起始的行为二级标题，以`###`起始的行为三级标题。以此类推。
# 一级标题
## 二级标题
### 二级标题

### 换行
使用空白行进行换行。

或者使用两个空格加回车换行。

### 等宽字体
用单引号包裹的文字显示为等宽字体：`text`

### 斜体

用星号包裹的文字`*text*` 显示为斜体：*text*

### 粗体
用双星号包裹的文字`**text**` 显示为粗体：**text**

### URL链接
URL链接可以写为`[链接文字](http://www.example.com)`: [链接文字](http://www.example.com) 

### 抄录环境(verbatim)
以4个空格起始的文字显示为无视格式符的抄录(verbatim)：

    def func(x):
        return x ** 2
        
### 表格
表格的输入格式如下：

    | A | B | C |
    |---|---|---|
    | 1 | 2 | 3 |
    | 4 | 5 | 6 |

| A | B | C |
|---|---|---|
| 1 | 2 | 3 |
| 4 | 5 | 6 |

### 水平分隔线

三个破折号`---`单独成为水平分隔线：

---

### 块引用

大于号 `>` 起始的行显示为块引用：
> Text here is indented and offset
> from the main text body.

### 无序列表
星号 `*` 起始的行为无序列表：

* Item one
* Item two
* Item three

### 有序列表
数字 `1.` `2.`等起始的行为有序列表：

1. Item one
2. Item two
3. Item three

### 图像
插入本地图像 `![Alternative text](image-file.png)`  
插入远程图像 `![Alternative text](http://www.example.com/image.png)`  
或者插入`HTML`代码段 `<img src="image-file.png" style="width: 80%;"/>`

### LaTeX数学公式
行中的 LaTeX 公式使用`$`号包裹。 `$\LaTeX$`显示为 $\LaTeX$  
单独居中显示的 LaTeX 公式使用`$$`号包裹。`$$\LaTeX$$`显示为
$$\LaTeX$$ 
也可以直接使用 latex 环境，例如equation, eqnarray, cases：

`\begin{equation} x = 1 \end{equation}`

\begin{equation} x = 1 \end{equation}

`\begin{eqnarray} x = 2 \end{eqnarray}`

\begin{eqnarray} x = 2 \end{eqnarray}

`\begin{cases}
x = 3 \\
y = 4 \\
\end{cases}`

\begin{cases} 
x = 3 \\
y = 4 \\
\end{cases}