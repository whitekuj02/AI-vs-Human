# AI-vs-Human
Ai vs human 데이콘 3일 헤커톤

최고 기록 accuary = 0.57

augmentationByGPT.py => train 데이터 셋을 비슷하게 GPT 한테 시킴

맞춤법이 핵심이었다...
그리고 
tokenizer = AutoTokenizer.from_pretrained('skt/kogpt2-base-v2')
model = AutoModel.from_pretrained('skt/kogpt2-base-v2')

이 모델을 사용할 것
