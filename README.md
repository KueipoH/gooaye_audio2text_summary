# 用 AI 輕鬆搞定股癌節目摘要 — Whisper & GPT 結合的財經資訊自動提取工具

在資訊爆炸的時代，如何在海量的財經內容中快速掌握重點？本專案正是為了解決這個問題而誕生——結合 OpenAI 的 Whisper 模型與 GPT API，自動將股癌（熱門財經 Podcast）或 YouTube 影片的音訊轉錄為文字，並智能提取所有關鍵的股票與產業資訊，讓你秒懂財經重點！

## 專案亮點

- **自動化音訊轉錄**  
  利用 Whisper 模型，將影片中音訊精準轉為文字，支援中文語音辨識，搭配初始提示提升識別準確度。

- **智能資訊提取**  
  結合 GPT 模型，從轉錄文字中一步步萃取股票代號、公司名稱及產業重點，讓資訊抓取不再繁瑣。

- **GPU 加速**  
  檢查並利用 GPU（若有）大幅提升處理速度，保證效率與準確度兼具。

- **結果格式化實際案例展示**
  - 假設我們處理「股癌」第537集，轉錄後的分析結果可能是這樣的：

```
第一步：整理「股票資訊」
下列為原文中明確出現、且具備「公司名稱」與「可能對應之代號」的股票/ETF，並將多次提及的內容合併為一條，盡量保留關鍵用詞與重點描述。

2330
2330 台積電（TSMC）

提及內容摘要：
「接下來我們進入市場話題，首先先聊一下台積電對美投資1000億…在市場上在Hedge Fund圈子傳的rumor…大家其實認為說還有一個Part 2…」
提到「台積電在美國設廠、Arizona工廠、2030年量產」
提到「不一定要投資Intel；若只是擴大投資美國，掛上‘1000億投資’有轉圜空間」
提到「晶片發展補助、美國的CHIP Act，以及台積電如何在各國佈局」
強調「研發中心（RD Center）不一定會把核心技術移走，且建廠是長期工程，短時間不會大幅影響台灣本部。」
𝐼𝑁𝑇𝐶
INTC Intel

提及內容摘要：
文中提到「市場傳言說台積電可能幫Intel投資或管理工廠」
「對Intel來說，如果取消CHIP Act補助，可能衝擊更大；Intel內部人士在LinkedIn上發言表明他們對自己量產能力很有自信。」
主要是謠言或傳聞層面，尚未有實質合作方案落定。
𝑁𝑉𝐷𝐴
NVDA NVIDIA

提及內容摘要：
「NVIDIA是CoreWeave的‘親兒子’，投資6%股份…CoreWeave也常在第一時間拿到最多算力與新產品（如GH200）…」
「AI需求暫時高漲，但若未來供需趨於平衡或出現更先進新產品，可能影響NVIDIA資料中心GPU銷售。」
也提到NVIDIA在台灣擴大研發中心。
CoreWeave（目前非上市公司，無股票代號）

提及內容摘要：
「CoreWeave過去兩年營收成長1300%、700%，最新一年約19億美元營收；但淨虧損也很大，最新一年虧損8.63億美元。」
「NVIDIA持股6%，CoreWeave以抵押GPU的方式舉債；若未來H100等產品折舊速度加快，財務壓力將增大。」
「即將IPO，傳估值欲達35B美元；可能面臨供需失衡的風險，也可能對整體AI算力租賃市場造成示範性影響。」
𝑇𝑆𝐿𝐴
TSLA Tesla

提及內容摘要：
「文中對特斯拉長期看法：純電動車銷售只是其基本面，更重要的看點在FSD（全自動駕駛）訂閱、Robotaxi與機器人等未來發展。」
也提到「股價在高點時都是好消息，低點時壞消息滿天飛；若想買便宜，通常都伴隨市場對該公司許多負面解讀。」
𝐴𝑀𝐷
AMD Advanced Micro Devices

提及內容摘要：
「與NVIDIA類似，都在台灣設有研發中心…」
內容提及AMD有在台灣布局，但無深入分析其近期股價或合作情形。
2317
2317 鴻海（Foxconn）

提及內容摘要：
「最有名的是鴻海，2016年到2018年間也曾承諾對美國投資，但最後因各種因素‘跳恰恰’，被外界認為是開了空頭支票…」
話題聚焦在過往承諾對美投資的落差，以及當時政治氛圍下的應對手法。
0050（元大台灣50 ETF）

提及內容摘要：
「有提到0050今年開始拆股；但不影響其追蹤台灣50大企業的本質…」
文中指出「若無法專心選股，可考慮0050之類ETF；報酬可長期超越多數投資人。」
00608（富邦台50 ETF）

提及內容摘要：
「同樣是追蹤台灣50指數的ETF，與0050為主流產品…長期投資邏輯與0050接近。」
ServiceNow（NOW）、Microsoft（MSFT）、CrowdStrike（CRWD）、Zscaler（ZS）、Salesforce（CRM）

提及內容摘要：
這些公司在文中零星被提到，多是舉例在「企業IT採用」角度：
「公司若決定採用某產品，通常能見度高、後續表現也不差，例如Now、Microsoft、Crowdstrike…」
「Zscaler被提到似乎表現較弱。」
僅屬於對部分SaaS/IT服務商的簡短評論。
（可能）
𝑇𝑋𝑁
TXN Texas Instruments
提及內容摘要：
文字中出現「一家與德儀競爭的IC公司面試邀約」，德儀即Texas Instruments；但該應徵公司並未公開明確名稱。
主要涉及「車用或工控半導體」的市場需求，文中推測今年在此領域仍有成長空間，但人員凍結（hiring freeze）也可能發生。
第二步：整理「產業資訊」
半導體產業

相關描述或分析重點：
「台積電、Intel、AMD、NVIDIA…皆涉及先進製程、全球佈局、美國CHIP Act補貼、供需變化…」
「投資美國是長期工程；可能有政治與供應鏈因素壓力。
「各家設立研發中心的位置、如何保留核心技術，是市場關注焦點。」
「市場盛傳台積電曾被施壓投資Intel或合作，但目前狀況偏向自行擴大投資設廠。」
AI/雲端算力租賃（CSP、GPU租賃）

相關描述或分析重點：
「CoreWeave等二線CSP陸續崛起，但財務壓力大；高額虧損及舉債購買GPU，將來折舊或價格下跌風險高。」
「NVIDIA、AWS、Azure、Google等一線雲端也在提供租賃；若供需漸趨平衡，恐出現價格競爭。」
「台灣若想切入算力租賃，可能面臨更激烈競爭和財務風險。」
電動車/車用產業

相關描述或分析重點：
「特斯拉除了賣車，更重點在FSD（全自動駕駛）訂閱、Robotaxi、機器人等未來題材。」
「車用IC、工業IC需求仍在，與德儀（TI）等相關業者競爭激烈。」
「電動車龍頭股價波動大，壞消息與好消息都會放大；投資人關注長期科技研發方向。」
第三步：補充「其他重要市場資訊」
關稅與補貼：

美國前任與現任政府都曾透過關稅或補助（如CHIP Act）來推動或施壓各國企業投資美國；台積電1000億美元的投資承諾，也被視為一種政治談判結果。
若CHIP Act補助被削減，對Intel此類公司恐造成相對更大的衝擊。
IPO與財務風險：

CoreWeave 擬以35B美元估值IPO，但因虧損規模龐大且抵押GPU舉債，未來若AI供需趨於平衡、GPU卡價跌或有新產品替代，可能引發債務與估值調整風險。
ETF長期策略：

文字中也提及「0050」「00608」等台股ETF，強調「ETF追蹤指數、長期買進持有」仍為一般投資人的選擇之一；短期拆股並不影響其投資本質。
投資心法：

原文多次強調「投資中滿街好消息或壞消息，常是股價已先行反映；真正的難處在於停利、停損與策略堅持。」
總結
根據原文內容，主要出現的個股及ETF包括台積電（2330/TSM）、Intel（INTC）、NVIDIA（NVDA）、CoreWeave（非上市）、Tesla（TSLA）、AMD（AMD）、鴻海（2317）、台灣50相關ETF（0050/00608）、以及若干美股SaaS/IT服務類（ServiceNow、Microsoft、CrowdStrike、Zscaler、Salesforce）等。產業方面以「半導體」、「AI/雲端算力租賃」、「電動車/車用IC」為核心，並延伸討論CHIP Act、全球佈局、IPO財務風險及ETF等投資策略。以上即為文字中明確且重點式的股票與產業資訊整理。
  ```

## 參考閱讀：用 AI 輕鬆搞定股票節目摘要

若您希望進一步了解如何運用 Whisper 模型與 GPT API 從音訊中自動萃取股票及產業資訊，歡迎參考我的 [Medium 文章](https://medium.com/@kueipo/%E7%94%A8ai%E8%BC%95%E9%AC%86%E6%90%9E%E5%AE%9A%E8%82%A1%E7%A5%A8%E7%AF%80%E7%9B%AE%E6%91%98%E8%A6%81-whisper%E6%A8%A1%E5%9E%8B%E8%AE%93%E4%BD%A0%E7%A7%92%E6%87%82-%E8%82%A1%E7%99%8C-%E7%B2%BE%E8%8F%AF%E5%9C%A8%E9%80%99%E7%AF%87%E6%96%87%E7%AB%A0-%E6%88%91%E5%B0%87-step-by-step-%E5%B8%B6%E4%BD%A0-53036ead5890)。  
在文章中，我將逐步帶您深入探討從音訊轉錄到重點資訊提取的全流程，讓您全面掌握 AI 在財經內容處理上的無限潛能。



