import torch
from transformers import BertTokenizer

from classification import LongformerClassifier

"""
初始化设备（根据是否可用选择CUDA或CPU）。

使用指定目录中的第一个.ckpt文件来加载Longformer分类器模型。
对一个长文本进行分词和编码，准备模型输入。
在指定设备上运行模型，对输入文本进行分类，并输出预测的类别。
"""

# 初始化设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 创建一个长输入文档
text = """《文明5》迎来重大更新 修正平衡性
　　《文明5》开发单位Firaxis Games宣布该作迎来重大升级，玩家重启Steam客户端后游戏将自动更新，最新版本为1.0.0.62：
　　其主要内容包括：
　　控制界面
　　修正新收的傀儡城可能会执着地要求玩家建造单位使本回合无法结束
　　飞机移库后数字现能正确反映在新址上
　　点选大军后资源图标不再消失
　　玩家现可决定是否自动轮转单位
　　经济纵览中可看到更详细的财政数据，比如各城市的产值，建筑维护费用等
　　经济纵览中可看到商路信息
　　经济纵览中新加入“资源及幸福指数(笑脸)”条目
　　在兼并/傀儡/烧城选择项中提供相应的红脸数据
　　若奇迹所需常规建筑数少于五座，则游戏会指出哪些城市缺少该类建筑
　　外交纵览中的全球政治形势现能提供更清晰的描述
　　Mod浏览及安装(略)
　　游戏
　　新增一个选项，使工人在自主状态下不会擅自更改玩家对此地块的设定
　　修正遣散某个单位后无法即时看到经济收入，直至遣散下一个单位时才看到有钱进帐的错误
　　遣散某单位只能获取该单位造价的10%
　　城防增加25%
　　商路形成及联通判定有多处修正
　　玩家现可出售城市中的建筑
　　玩家现可向城邦赠送飞行单位
　　单位晋级选项中的“医疗”现只能治疗邻近单位
　　修正以弓为武器的步射及骑射单位的晋级项
　　涉渡单位现无法阻碍对手的地面单位
　　改进单位轮转视角，缓解镜头的跳跃感
　　AI
　　较弱的对手更倾向于不接受或不发起边界开放协议
　　停战协定或科研协定永不到期的错误已被修正
　　限制城市市民增长的可选项终于实现其既定功能
　　工人在自主状态下喜欢造贸易站的势头得以遏制，并重新调整了各建造选项的优先级
　　傀儡城市不再建造耗费资源(Resources，可能单指石油/煤炭/铝/铁这些战略资源而非维护费用)的建筑
　　可勒令傀儡城市节省一切不必要开销以专注于提供财政支持
　　AI对手在自己的城市受到威胁时可更自如地利用驻军及周边地利
　　多人模式
　　修正了馈赠漏洞
　　改进了文字聊天界面，各人发言可由颜色区分并带有音效提醒
　　在交易生效前有附加的审核手续以确保条款无误
　　杂项
　　宣战后科研协定作废，玩家不再得到科研成果
　　修正了某些状况下存档失效的错误
　　修正了某些运行环境下加载巨型地图可能使游戏崩溃的错误
　　修正了使用某些显卡时导致画面不正常的错误
　　修正了游戏刷新画面时使用低分辨率地形的错误
　　修正了后台渲染游戏单位导致游戏崩溃的错误
　　修正了多处游戏崩溃错误
　　修正了宽高比小于1的地图会导致游戏崩溃的错误
　　游戏教程多处修正

"""

# 获取第一个.ckpt文件路径
fpath = 'save\\version_0\\checkpoints\\ep-epoch=0_acc-acc=0.921.ckpt'

# 加载模型和分词器
model = LongformerClassifier.load_from_checkpoint(fpath)
tokenizer = BertTokenizer.from_pretrained('D:\\models\\longformer-chinese-base-4096')

# 对输入文本进行分词和编码
tokenized = tokenizer.encode_plus(
            text=text,
            return_tensors='pt',
            padding='max_length',
            max_length=512,
            truncation=True,
            add_special_tokens=True
        )
input_ids = tokenized['input_ids']
attention_mask = tokenized['attention_mask']

# 将模型和输入数据移动到指定设备上
model.to(device)
input_ids = input_ids.to(device)
attention_mask = attention_mask.to(device)

# 设置模型为评估模式
model.eval()

# 运行模型并获取预测结果
outputs = model(input_ids, attention_mask)
logits, _ = outputs
predicted_class = torch.argmax(logits, dim=1).item()

# 打印预测类别
print(predicted_class)