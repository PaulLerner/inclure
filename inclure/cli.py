# coding: utf-8
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer

from jsonargparse import CLI


def pipeline(text, model, tokenizer):
    inputs = tokenizer([text], return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=model.config.max_length)
    output = tokenizer.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return output
    

def user_loop(model, tokenizer, message):
    while True:
        answer = input(f"{message}\n>>> ").strip()
   #      if answer.lower() == "q":
 #            break
        output = pipeline(answer,model, tokenizer)[0]
        print(f"{output}\n")
        

def main(inclure: bool = True):
    if inclure:
        model_name = "PaulLerner/fabien.ne_barthez"
        message = "Je suis Fabien.ne BARThez. Écris ta phrase en français standard pour que je la traduise en français inclusif !"
    else:
        model_name = "PaulLerner/fabien_barthez"
        message = "Je suis Fabien BARThez. Écris ta phrase en français inclusif pour que je la traduise en français standard !"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    user_loop(model, tokenizer, message)
        
    
if __name__ == '__main__':
    CLI(main)
