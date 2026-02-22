import os
import torch
import torch.nn as nn
import random
import numpy as np
# import matplotlib.pyplot as plt  # [新增] 引入画图库
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    GPT2LMHeadModel,
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling,
    GPT2Config,
    TrainerCallback  # [新增] 引入回调类
)
from tqdm import tqdm

# ===========================
# 0. 全局配置
# ===========================
MAX_LENGTH = 128
EVAL_INTERVAL = 10  # [新增] 每多少步评估一次

QUESTION_TEMPLATES = [
    "Question: A farmer has {a} apples and buys another {b} ones, now how many apples he has in total? Answer: {s}",
    "Question: There are {a} birds on the tree, and {b} more fly in. How many birds are there now? Answer: {s}",
    "Question: Tom has {a} books. His mother gives him {b} more. What is the total number of books? Answer: {s}",
    "Question: A box contains {a} red balls and {b} blue balls. How many balls are in the box? Answer: {s}",
    "Question: Alice saved {a} dollars last week and {b} dollars this week. How much did she save in total? Answer: {s}",
    "Question: The class has {a} boys and {b} girls. What is the total number of students? Answer: {s}",
    "Question: A parking lot has {a} cars. {b} more cars park there. How many cars are there? Answer: {s}",
    "Question: Chef cooked {a} steaks and {b} burgers. How many dishes did he cook? Answer: {s}",
    "Question: I read {a} pages yesterday and {b} pages today. How many pages did I read? Answer: {s}",
    "Question: The store sold {a} pencils in the morning and {b} in the afternoon. Total pencils sold? Answer: {s}"
]

# ===========================
# 1. 自定义模型结构
# ===========================

class CustomGPT2Model(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        self.condition_projector = nn.Linear(config.n_embd, config.n_embd, bias=False)
        nn.init.normal_(self.condition_projector.weight, std=0.02)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        segment_ids=None, 
        labels=None,
        **kwargs
    ):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Step A: 构建非对称 Attention Mask
        indices = torch.arange(seq_len, device=device)
        causal_mask = indices.unsqueeze(1) >= indices.unsqueeze(0) 
        causal_mask = causal_mask.view(1, 1, seq_len, seq_len)
        
        if segment_ids is not None:
            seg_i = segment_ids.unsqueeze(2)
            seg_j = segment_ids.unsqueeze(1)
            
            # Condition (0) 可以看 Question (1)
            can_see_future = (seg_i == 0) & (seg_j == 1)
            # Question/Answer (>=1) 不能看 Condition (0)
            must_block_past = (seg_i >= 1) & (seg_j == 0)
            
            final_bool_mask = (causal_mask | can_see_future.unsqueeze(1)) & (~must_block_past.unsqueeze(1))
        else:
            final_bool_mask = causal_mask

        if attention_mask is not None:
            padding_mask = attention_mask.view(batch_size, 1, 1, seq_len).bool()
            final_bool_mask = final_bool_mask & padding_mask

        extended_attention_mask = torch.zeros_like(final_bool_mask, dtype=torch.float)
        extended_attention_mask = extended_attention_mask.masked_fill(~final_bool_mask, torch.finfo(self.dtype).min)

        # Step B: Backbone Forward
        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=extended_attention_mask,
            **kwargs
        )
        hidden_states = transformer_outputs[0]

        # Step C: Logits
        lm_logits = self.lm_head(hidden_states)
        final_logits = lm_logits.clone()
        
        # Step D: L2 Logic
        if segment_ids is not None:
            projected_context = self.condition_projector(hidden_states)

            for b in range(batch_size):
                cond_indices = (segment_ids[b] == 0).nonzero(as_tuple=True)[0]
                if len(cond_indices) == 0: continue
                
                if labels is not None:
                    valid_label_indices = (labels[b] != -100).nonzero(as_tuple=True)[0]
                    if len(valid_label_indices) > 0:
                        q_idx = valid_label_indices[0] - 1
                    else: q_idx = -1
                else:
                    if attention_mask is not None:
                        q_idx = int(attention_mask[b].sum().item()) - 2
                    else:
                        q_idx = seq_len - 1

                if q_idx < 0 or q_idx >= seq_len: continue

                query_vec = hidden_states[b, q_idx, :].unsqueeze(0)
                key_vecs = projected_context[b, cond_indices, :]
                scores = torch.matmul(query_vec, key_vecs.transpose(0, 1))
                cond_token_ids = input_ids[b, cond_indices]
                unique_tokens, _ = torch.unique(cond_token_ids, return_inverse=True)
                
                for token_id in unique_tokens:
                    locs = (cond_token_ids == token_id).nonzero(as_tuple=True)[0]
                    max_score = scores[0, locs].max()
                    final_logits[b, q_idx, token_id] = max_score

        # Step E: Loss
        loss = None
        if labels is not None:
            shift_logits = final_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # 移除原来的 print debug，避免训练过程中刷屏，影响进度条显示
            # 如果需要调试，可以在 custom_evaluation 里看

        return (loss, final_logits) if loss is not None else (final_logits,)

# ===========================
# 2. 数据集构造 (保持不变)
# ===========================

class ArithmeticDataset(Dataset):
    def __init__(self, tokenizer, num_samples, method='CoQE', is_train=True, is_distorted=False):
        self.tokenizer = tokenizer
        self.data = []
        self.method = method
        min_val, max_val = 0, 45 
        letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        for _ in range(num_samples):
            cond_pairs = []
            for _ in range(3):
                a, b = random.randint(min_val, max_val), random.randint(min_val, max_val)
                cond_pairs.append((a, b, a + b))
            
            target_idx = random.randint(0, 2)
            if is_distorted:
                cond_pairs[target_idx] = random.sample(letters, 3)
            t_a, t_b, t_s_real = cond_pairs[target_idx]
            
            cond_strs = [f"{a} + {b} = {s}" for a, b, s in cond_pairs]
            expected_s = t_s_real
            condition_text = "Condition: " + ". ".join(cond_strs) + "."
            
            template = random.choice(QUESTION_TEMPLATES)
            temp_full = template.format(a=t_a, b=t_b, s=expected_s)
            q_part_str, a_part_str = temp_full.split("Answer:")
            
            question_prompt_text = " " + q_part_str + "Answer:"
            answer_text = " " + str(expected_s) 
            full_prompt = condition_text + question_prompt_text
            
            self.data.append({
                "condition_text": condition_text,
                "question_prompt_text": question_prompt_text,
                "answer_text": answer_text,
                "prompt": full_prompt,
                "target_val": str(expected_s)
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        cond_tokens = self.tokenizer(item["condition_text"], add_special_tokens=False)["input_ids"]
        q_prompt_tokens = self.tokenizer(item["question_prompt_text"], add_special_tokens=False)["input_ids"]
        ans_tokens = self.tokenizer(item["answer_text"], add_special_tokens=False)["input_ids"]
        
        input_ids = cond_tokens + q_prompt_tokens + ans_tokens
        segment_ids = [0]*len(cond_tokens) + [1]*len(q_prompt_tokens) + [2]*len(ans_tokens)

        if self.method == 'CoQE':
            labels = [-100] * (len(cond_tokens) + len(q_prompt_tokens)) + ans_tokens
        elif self.method == 'Transformer':
            labels = cond_tokens + q_prompt_tokens + ans_tokens
        
        cur_len = len(input_ids)
        pad_len = MAX_LENGTH - cur_len
        attention_mask = [1] * cur_len
        
        if pad_len > 0:
            input_ids += [self.tokenizer.pad_token_id] * pad_len
            segment_ids += [2] * pad_len 
            labels += [-100] * pad_len
            attention_mask += [0] * pad_len
        else:
            input_ids = input_ids[-MAX_LENGTH:]
            segment_ids = segment_ids[-MAX_LENGTH:]
            labels = labels[-MAX_LENGTH:]
            attention_mask = attention_mask[-MAX_LENGTH:]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "segment_ids": torch.tensor(segment_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "meta_prompt": item["prompt"],
            "meta_target": item["target_val"]
        }

# ===========================
# 3. 评估函数
# ===========================

def custom_evaluation(model, tokenizer, eval_dataset, device, batch_size=32, desc="Evaluating", silent=False):
    """
    silent=True 时减少打印输出
    """
    model.eval()
    correct = 0
    total = 0
    
    def collate_fn(batch):
        input_ids = torch.stack([b['input_ids'] for b in batch])
        att_mask = torch.stack([b['attention_mask'] for b in batch])
        seg_ids = torch.stack([b['segment_ids'] for b in batch])
        meta_prompts = [b['meta_prompt'] for b in batch]
        meta_targets = [b['meta_target'] for b in batch]
        return {
            "input_ids": input_ids,
            "attention_mask": att_mask,
            "segment_ids": seg_ids,
            "meta_prompts": meta_prompts,
            "meta_targets": meta_targets
        }

    loader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # 如果是 silent 模式，就不显示 tqdm，或者简化显示
    if not silent:
        print(f"\nStart Evaluation: {desc}")
        iterator = tqdm(loader)
    else:
        iterator = loader

    with torch.no_grad():
        for batch in iterator:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            segment_ids = batch["segment_ids"].to(device)
            prompts = batch["meta_prompts"]
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                segment_ids=segment_ids
            )
            logits = outputs[0]
            
            for i in range(len(prompts)):
                valid_len = int(attention_mask[i].sum().item())
                pred_pos = valid_len - 2
                target_pos = valid_len - 1
                pred_logits = logits[i, pred_pos, :]
                pred_id = torch.argmax(pred_logits).item()
                target_id = input_ids[i, target_pos].item()
                
                if pred_id == target_id:
                    correct += 1
                total += 1

    acc = correct / total if total > 0 else 0
    if not silent:
        print(f"[{desc}] Top-1 Accuracy: {acc:.4f} ({correct}/{total})")
    return acc

# ===========================
# 4. 新增 Callback 用于过程评估
# ===========================

class EvalLoggingCallback(TrainerCallback):
    def __init__(self, model, tokenizer, iwl_dataset, icl_dataset, device):
        self.model = model
        self.tokenizer = tokenizer
        self.iwl_dataset = iwl_dataset
        self.icl_dataset = icl_dataset
        self.device = device
        
        # 记录数据的容器
        self.history = {
            "step": [],
            "iwl_acc": [],
            "icl_acc": []
        }

    def on_step_end(self, args, state, control, **kwargs):
        # 判断是否达到指定的步数间隔
        if state.global_step > 0 and state.global_step % EVAL_INTERVAL == 0:
            current_step = state.global_step
            
            # 临时打印
            print(f"\n[Step {current_step}] Performing intermediate evaluation...")
            
            # 运行评估 (Silent模式，避免刷屏)
            iwl_acc = custom_evaluation(
                self.model, self.tokenizer, self.iwl_dataset, self.device, 
                desc="Step IWL", silent=True
            )
            icl_acc = custom_evaluation(
                self.model, self.tokenizer, self.icl_dataset, self.device, 
                desc="Step ICL", silent=True
            )
            
            # 记录数据
            self.history["step"].append(current_step)
            self.history["iwl_acc"].append(iwl_acc)
            self.history["icl_acc"].append(icl_acc)
            
            print(f"  -> IWL Acc: {iwl_acc:.2%} | ICL Acc: {icl_acc:.2%}")
            
            # 评估完后确保模型切回训练模式
            self.model.train()

# ===========================
# 5. 主流程
# ===========================
def main():
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    device = torch.device(f"cuda:{local_rank}" if local_rank != -1 else "cuda" if torch.cuda.is_available() else "cpu")

    # 模型路径
    MODEL_NAME = "/GPT2-Medium"
    METHOD = 'CoQE' # or "Transformer"
    SEED = 42 + local_rank
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Datasets
    print("Generating Datasets...")
    train_dataset = ArithmeticDataset(tokenizer, num_samples=5000, method=METHOD, is_train=True, is_distorted=False)
    # 测试集稍微小一点，为了评估速度快一点
    test_iwl_dataset = ArithmeticDataset(tokenizer, num_samples=200, method=METHOD, is_train=False, is_distorted=False)
    test_icl_dataset = ArithmeticDataset(tokenizer, num_samples=200, method=METHOD, is_train=False, is_distorted=True)

    # 3. Custom Model (恢复使用 CustomGPT2Model)
    print(f"Loading Custom Model based on {MODEL_NAME}...")
    config = GPT2Config.from_pretrained(MODEL_NAME)
    
    if METHOD == 'CoQE':
        model = CustomGPT2Model.from_pretrained(MODEL_NAME, config=config)
    elif METHOD == 'Transformer':
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, config=config) # 原代码注释掉了
    
    model.tokenizer = tokenizer
    model.to(device)
    
    # 4. Zero-shot Evaluation (Baseline)
    acc_zero_shot_iwl = custom_evaluation(model, tokenizer, test_iwl_dataset, device, desc="Zero-shot IWL")
    acc_zero_shot_icl = custom_evaluation(model, tokenizer, test_icl_dataset, device, desc="Zero-shot ICL")
    
    # 5. Training
    print("\n>>> Starting Finetuning...")
    
    training_args = TrainingArguments(
        output_dir="./custom_math_icl_results",
        overwrite_output_dir=True,
        num_train_epochs=2,  # 增加一点 epoch 保证 step 数足够
        per_device_train_batch_size=16, # 调小batch size 以增加 step 数量
        learning_rate=5e-5,
        save_strategy="no",
        logging_steps=10, 
        report_to="none",
        remove_unused_columns=False, # 必须保留 segment_ids
        weight_decay=5e-2 
    )

    def data_collator(features):
        batch = {
            "input_ids": torch.stack([f["input_ids"] for f in features]),
            "attention_mask": torch.stack([f["attention_mask"] for f in features]),
            "segment_ids": torch.stack([f["segment_ids"] for f in features]),
            "labels": torch.stack([f["labels"] for f in features]),
        }
        return batch

    # 实例化 Callback
    eval_callback = EvalLoggingCallback(model, tokenizer, test_iwl_dataset, test_icl_dataset, device)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=[eval_callback] # 添加 callback
    )
    
    # 记录 Step 0 (Zero-shot)
    eval_callback.history["step"].append(0)
    eval_callback.history["iwl_acc"].append(acc_zero_shot_iwl)
    eval_callback.history["icl_acc"].append(acc_zero_shot_icl)
    
    trainer.train()
    
    print("-"*50)
    print("iwl_scores")
    print(eval_callback.history["iwl_acc"])    
    print("icl_scores")
    print(eval_callback.history["icl_acc"])


if __name__ == "__main__":
    main()