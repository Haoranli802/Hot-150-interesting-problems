# Hot-150-interesting-problems

## 数组 / 字符串

### LC88合并两个有序数组：

用两个指针指向两个数组的末尾位置，然后再设置一个指针指向结果数组的末尾，然后当两个数组指针都不为0的时候，对比当前数字大小加入结果数组并且缩短对应指针和总指针，然后最后判断数组二是否有剩余，如果有的话再把数组二中剩余加入结果数组。

***

### LC380. O(1) 时间插入、删除和获取随机元素:

用一个长数组记录数字，一个Random实例来实现随机数，一个哈希表来记录数字的位置，一个index来记录当前最后一个数字的位置，初始化为1。insert：先判断是不是已有数据，如果没有，先加入，然后记录位置。delete：先判断是不是已有数据，如果是，判断是不是最后一个数字，如果是的话，直接删掉，index-1。如果不是，那么把要删除的数字和末尾的数字交换，然后删除，index-1。random：直接返回数组中的随机数据即可。使用random.nextInt(index+1)

***

### LC13. 罗马数字转整数

首先可以创建一个罗马数字转数字的函数，然后设置String的第一个字符为pre字符，遍历String，如果pre < cur 那么代表有一个小的数字在大的数字前面，那么sum -= pre，如果不小于，那么sum += pre，遍历完记得加上最后一位数字。

***

### LC68. 文本左右对齐

首先设置count 和起始点 index，然后遍历字符串列表，如果当前字符串是最后一个或者count加上下一个字符串超过限制的话，加上目前遍历过的字符串。加上的步骤为，首先计算当前包含的字符串数量，然后结算出平均空格数和空余空格数，然后把空余空格加到一个字符串后面，然后加上其他字符串即可。如果是最后一行，那么每个字符串中间一定只隔一个空格，然后把剩余的全部加到尾部。

```
class Solution {
        public List<String> fullJustify(String[] words, int maxWidth) {
            List<String> res = new ArrayList<>();
            int cnt = 0, bg = 0;
            for (int i = 0; i < words.length; i++) {
                cnt += words[i].length() + 1;
                // 如果是最后一个单词，或者加上下一个词就超过长度了，即可凑成一行
                if (i + 1 == words.length || cnt + words[i + 1].length() > maxWidth) {
                    // 对每行单词进行空格平均划分
                    res.add(fillWords(words, bg, i, maxWidth, i + 1 == words.length));
                    bg = i + 1;
                    cnt = 0;
                }
            }
            return res;
        }

        /**
         * 对每行单词进行空格平均划分
         */
        private String fillWords(String[] words, int bg, int ed, int maxWidth, boolean lastLine) {
            int wordCount = ed - bg + 1;
            // 除去每个单词尾部空格， + 1 是最后一个单词的尾部空格的特殊处理
            int spaceCount = maxWidth + 1 - wordCount;
            for (int i = bg; i <= ed; i++) {
                // 除去所有单词的长度
                spaceCount -= words[i].length();
            }
            // 词尾空格
            int spaceSuffix = 1;
            // 额外空格的平均值 = 总空格数/间隙数
            int spaceAvg = (wordCount == 1) ? 1 : spaceCount / (wordCount - 1);
            // 额外空格的余数
            int spaceExtra = (wordCount == 1) ? 0 : spaceCount % (wordCount - 1);
            // 填入单词
            StringBuilder sb = new StringBuilder();
            for (int i = bg; i < ed; i++) {
                sb.append(words[i]);
                if (lastLine) {
                    sb.append(" ");
                    continue;
                }
                int n = spaceSuffix + spaceAvg + (((i - bg) < spaceExtra) ? 1 : 0);
                while (n-- > 0) {
                    sb.append(" ");
                }
            }
            // 填入最后一个单词
            sb.append(words[ed]);
            // 补上这一行最后的空格
            int lastSpaceCnt = maxWidth - sb.length();
            while (lastSpaceCnt -- > 0){
                sb.append(" ");
            }
            return sb.toString();
        }
    }
}
```

### LC30. 串联所有单词的子串

```
class Solution {
    public static List<Integer> findSubstring(String s, String[] words) {
        List<Integer> res = new ArrayList<>();
        if (s == null || s.length() == 0 || words == null || words.length == 0) {
            return res;
        }
        int wordNum = words.length;
        int wordLen = words[0].length();
        // 将单词数组构建成哈希表
        Map<String, Integer> map = new HashMap<>();
        for (String word : words) {
            map.put(word, map.getOrDefault(word, 0) + 1);
        }
        // 这里只需遍历0~wordLen即可，因为滑动窗口都是按照wordLen的倍数进行滑动的
        for (int i = 0; i < wordLen; i++) {
            Map<String, Integer> tmp = new HashMap<>();
            // 滑动窗口
            int left = i, right = i, hit = 0;
            while (right + wordLen <= s.length()) {
                String word = s.substring(right, right + wordLen);
                right += wordLen;
                if (map.containsKey(word)) {
                    int num = tmp.getOrDefault(word, 0) + 1;
                    tmp.put(word, num);
                    hit++;
                    // 出现情况三，遇到了符合的单词，但是次数超了
                    if (map.get(word) < num) {
                        // 一直移除单词，直到次数符合
                        while (map.get(word) < tmp.get(word)) {
                            String deleteWord = s.substring(left, left + wordLen);
                            tmp.put(deleteWord, tmp.get(deleteWord) - 1);
                            left += wordLen;
                            hit--;
                        }
                    }
                } else {
                    // 出现情况二，遇到了不匹配的单词，直接将 left 移动到该单词的后边
                    tmp.clear();
                    hit = 0;
                    left = right;
                }
                if (hit == wordNum) {
                    res.add(left);
                    // 出现情况一，子串完全匹配，我们将上一个子串的第一个单词从tmp中移除，窗口后移wordLen
                    String firstWord = s.substring(left, left + wordLen);
                    tmp.put(firstWord, tmp.get(firstWord) - 1);
                    hit--;
                    left = left + wordLen;
                }
            }
        }
        return res;
    }
}
```

***

### LC399. 除法求值

```
class Solution {
    public double[] calcEquation(List<List<String>> equations, double[] values, List<List<String>> queries) {
        int nvars = 0;
        Map<String, Integer> variables = new HashMap<String, Integer>();

        int n = equations.size();
        for (int i = 0; i < n; i++) {
            if (!variables.containsKey(equations.get(i).get(0))) {
                variables.put(equations.get(i).get(0), nvars++);
            }
            if (!variables.containsKey(equations.get(i).get(1))) {
                variables.put(equations.get(i).get(1), nvars++);
            }
        }

        // 对于每个点，存储其直接连接到的所有点及对应的权值
        List<Pair>[] edges = new List[nvars];
        for (int i = 0; i < nvars; i++) {
            edges[i] = new ArrayList<Pair>();
        }
        for (int i = 0; i < n; i++) {
            int va = variables.get(equations.get(i).get(0)), vb = variables.get(equations.get(i).get(1));
            edges[va].add(new Pair(vb, values[i]));
            edges[vb].add(new Pair(va, 1.0 / values[i]));
        }

        int queriesCount = queries.size();
        double[] ret = new double[queriesCount];
        for (int i = 0; i < queriesCount; i++) {
            List<String> query = queries.get(i);
            double result = -1.0;
            if (variables.containsKey(query.get(0)) && variables.containsKey(query.get(1))) {
                int ia = variables.get(query.get(0)), ib = variables.get(query.get(1));
                if (ia == ib) {
                    result = 1.0;
                } else {
                    Queue<Integer> points = new LinkedList<Integer>();
                    points.offer(ia);
                    double[] ratios = new double[nvars];
                    Arrays.fill(ratios, -1.0);
                    ratios[ia] = 1.0;

                    while (!points.isEmpty() && ratios[ib] < 0) {
                        int x = points.poll();
                        for (Pair pair : edges[x]) {
                            int y = pair.index;
                            double val = pair.value;
                            if (ratios[y] < 0) {
                                ratios[y] = ratios[x] * val;
                                points.offer(y);
                            }
                        }
                    }
                    result = ratios[ib];
                }
            }
            ret[i] = result;
        }
        return ret;
    }
}

class Pair {
    int index;
    double value;

    Pair(int index, double value) {
        this.index = index;
        this.value = value;
    }
}
```

***

### LC909. 蛇梯棋

先扁平化board，然后设置queue来记录走到的地方，map来记录对应的step，首先先加入起始点(1, 0)，然后每一个点，都先判断是不是终点，如果是终点，那么返回对应的step，如果不是，那么尝试走1 ～ 6步，如果新的点越界，那么跳过，如果新的点有蛇或者梯子，变更点，如果新的点已经走过，跳过，然后把新的点加入queue，stpe + 1加入map，然后继续遍历直到找到终点，如果最后没找到终点，返回-1.

***

### LC433. 最小基因变化

现创建见一个常字符数组保存四个可替换的字母，然后把bank的String放入到set去重，然后创建map保存基因对应的step和负责防止重复加入，queue来保存转换后的基因，然后加入startGene到queue和map。然后层序遍历queue，对每一个字符串，转换为char[], 然后对每一个字符，尝试替换成一个不同的四字符，方法是先用char[].clone克隆原始字符组，然后替换。然后对于新的字符串，如果存在在set而不存在在map，判断是否是end，如果是返回当前step + 1，如果不是，加入queue，map加入对应step，继续遍历。如果遍历完没找到，返回-1.

***

### LC212. 单词搜索 II

```
class Solution {
    class TrieNode {
        String s;
        TrieNode[] tns = new TrieNode[26];
    }
    void insert(String s) {
        TrieNode p = root;
        for (int i = 0; i < s.length(); i++) {
            int u = s.charAt(i) - 'a';
            if (p.tns[u] == null) p.tns[u] = new TrieNode();
            p = p.tns[u];
        }
        p.s = s;
    }
    Set<String> set = new HashSet<>();
    char[][] board;
    int n, m;
    TrieNode root = new TrieNode();
    int[][] dirs = new int[][]{{1,0},{-1,0},{0,1},{0,-1}};
    boolean[][] vis = new boolean[15][15];
    public List<String> findWords(char[][] _board, String[] words) {
        board = _board;
        m = board.length; n = board[0].length;
        for (String w : words) insert(w);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                int u = board[i][j] - 'a';
                if (root.tns[u] != null) {
                    vis[i][j] = true;
                    dfs(i, j, root.tns[u]);
                    vis[i][j] = false;
                }
            }
        }
        List<String> ans = new ArrayList<>();
        for (String s : set) ans.add(s);
        return ans;
    }
    void dfs(int i, int j, TrieNode node) {
        if (node.s != null) set.add(node.s);
        for (int[] d : dirs) {
            int dx = i + d[0], dy = j + d[1];
            if (dx < 0 || dx >= m || dy < 0 || dy >= n) continue;
            if (vis[dx][dy]) continue;
            int u = board[dx][dy] - 'a';
            if (node.tns[u] != null) {
                vis[dx][dy] = true;
                dfs(dx, dy, node.tns[u]);
                vis[dx][dy] = false;
            }
        }
    }
}
```

***

### LC427. 建立四叉树

思路：建立helper function，遍历整个数组，判断整个数组是否相等，如果是的话，返回节点值和isLeaf，如果不是那么返回false，false和四个新的helper function的值。
```
    private Node dfs(int[][] grid, int rowS, int rowE, int colS, int colE){
       boolean same = true;
       int first = grid[rowS][colS];
       for(int i = rowS; i < rowE; i++){
            for(int j = colS; j < colE; j++){
                if(grid[i][j] != first) same = false;
            }
        }
        if(same) return new Node(first == 1, true);
        return new Node(false, false, 
        dfs(grid, rowS, (rowS + rowE) / 2, colS, (colS + colE) / 2),
        dfs(grid, rowS, (rowS + rowE) / 2, (colS + colE) / 2, colE),
        dfs(grid, (rowS + rowE) / 2, rowE, colS, (colS + colE) / 2),
        dfs(grid, (rowS + rowE) / 2, rowE, (colS + colE) / 2, colE)); 
    }
```

***

### LC162. 寻找峰值

思路：二分，不过要跟着数组上升的方向走，如果mid的值小于mid + 1，那么left = mid + 1，如果不小于的话，right = mid，最后返回left

***

### LC502. IPO

思路：先把profit跟capical绑定成数组，然后根据capital从小到大排序，然后创建priorityqueue和index = 0，进入while loop，条件为k--大于0，如果下标为i的数组capital 小于等于 w，那么加入priorityqueue然后i++，直到i == n或者w 小于当前capital，然后如果priorityqueue里面什么都没有，退出循环，如果有东西的话w加上队列头继续循环直到队列为空或者k == 0，然后返回w。

***

### LC373. 查找和最小的 K 对数字

思路：首先用长度为3的数组来表示一对数字，第一位为和，第二位为nums1下标，第三位为nums2下标。然后创建priorityqueue，队列比对逻辑为a[0] - b[0]。然后约定好，每拿出一组数字，只把  
(i, j + 1)加入到数组，因为(i, j)可以由(i - 1, j)和(i, j - 1)得出，那么会出现重复，所以只加入(i, j + 1)来得到(i, j)。假设nums1长度为n那么我们同时要把(0, 0)到(Math.min(n, k) - 1, 0)加入以便于计算。然后每次遍历priorityqueue时拿去堆顶部，然后加上(i, j + 1)直到满足k个即可。

***

## 位运算

### LC190. 颠倒二进制位

思路
```
十进制
ans = ans * 10 + n % 10;
n /= 10;

二进制
ans = ans * 2 + n % 2;
n /= 2;

位运算
ans = 0;
ans = (ans << 1) | (n & 1)  ans右移一位然后加上n的最后一位数
n = n >> 1; n左移一位，因为最后一位数已经被加到ans里面了

code
int res = 0;
for(int i = 0; i < 32; i++){  要走32位，因为前置0是有影响的
    res = (res << 1) | (n & 1);
    n = n >> 1;
}
return res;

Time: O(logN)
Space: O(1)
```
