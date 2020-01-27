import MeCab

mt = MeCab.Tagger("mecabrc")
str_in = "特急はくたかで富山に向かいます。それから、金沢に行って、兼六園に行きます。"
res = mt.parseToNode(str_in)

while res:
    arr = res.feature.split(",")
    if (arr[1] == "固有名詞"):
        print(res.feature)
        print(arr[6])
    res = res.next
