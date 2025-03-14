PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 lm_eval --model hf-multimodal \
--model_args pretrained=OpenGVLab/InternVL2_5-8B,dtype=bfloat16,trust_remote_code=True,convert_img_format=True,use_flash_attn=True,image_token_id=92544 \
--gen_kwargs "max_gen_toks=256,until=None" \
--device cuda --output_path="/home/jovyan/akharitonov/multimodal-harness/multimodal_results" --batch_size 1 \
--seed 1234 --log_samples \
--tasks mmmu_val --limit 1 \
--include_path "/home/jovyan/akharitonov/multimodal-harness/multimodal_tasks"
# The options below are unnessesary, since InternVL incorporates chat template internally
# --apply_chat_template --fewshot_as_multiturn --num_fewshot 0 