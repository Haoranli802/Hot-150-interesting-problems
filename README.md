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
