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
