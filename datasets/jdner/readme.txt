kb_jd_jsonl.txt 为京东sku数据，每行为一个sku的描述信息，jsonl格式，其中base_main_sku_id字段为商品唯一id，base_item_url为商品链接，其余为商品属性
kb_sn_jsonl.txt 为苏宁sku数据，格式与京东sku数据相同
train.txt, eval.txt, test.txt 分别为训练集、验证集、测试集，格式均为“京东sku_id	苏宁sku_id	是否匹配(1/0)”，中间以tab分割