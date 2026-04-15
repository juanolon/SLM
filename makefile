SHELL := /bin/bash

train_sudoku:
	python3 main.py \
		model=nano \
		data=sudoku \
		parameterization=new_diff \
		backbone=dit_bfn \
		model.length=81 \
		eval.compute_generative_perplexity=False \
		wandb.name=nano_sudoku \
		sampling.steps=1000 \
		trainer.val_check_interval=347 \
		training.beta_bfn=0.75 \
		checkpointing.resume_from_ckpt=False \
		T=1000 \
		loader.global_batch_size=128 \
		eval.new_diff_calculate=full \
		trainer.devices=1 \


train_text8:
	python3 main.py \
		model=small \
		data=text8 \
		parameterization=new_diff \
		backbone=dit_bfn \
		model.length=256 \
		eval.compute_generative_perplexity=False \
		wandb.name=tiny_text8_512 \
		sampling.steps=1000 \
		trainer.val_check_interval=347 \
		training.beta_bfn=0.75 \
		checkpointing.resume_from_ckpt=False \
		T=1000 \
		loader.global_batch_size=128 \
		eval.new_diff_calculate=full \
		trainer.devices=1 \


train_uniref50:
	python3 main.py \
		model=evodiff \
		data=uniref50 \
		parameterization=new_diff \
		backbone=dit_bfn \
		model.length=1024 \
		sampling.length=400 \
		eval.compute_generative_perplexity=False \
		wandb.name=tiny_uniref_evodiff \
		sampling.steps=500 \
		trainer.val_check_interval=500 \
		training.beta_bfn=0.75 \
		checkpointing.resume_from_ckpt=False \
		T=500 \
		loader.global_batch_size=128 \
		eval.new_diff_calculate=full \
		trainer.devices=1; \


sample_uniref50:
	$(eval CKPT_PATH:=/AIRvePFS/ai4science/users/yupei/test_slm_repo/ckpt/Uniref_protein.ckpt)
	$(eval OUTPUT_DIR:=/AIRvePFS/ai4science/users/yupei/test_slm_repo/output/output_protein)
	for sample_len in 100 200 300 400 500; do \
		python3 main.py \
			model=evodiff \
			data=uniref50 \
			mode=sample_eval \
			parameterization=new_diff \
			backbone=dit_bfn \
			model.length=1024 \
			sampling.length=$$sample_len \
			sampling.outdir=${OUTPUT_DIR} \
			eval.compute_generative_perplexity=False \
			wandb.name=sample_uniref_evodiff \
			sampling.steps=500 \
			trainer.val_check_interval=500 \
			training.beta_bfn=0.75 \
			checkpointing.resume_from_ckpt=True \
			T=500 \
			loader.global_batch_size=100 \
			eval.new_diff_calculate=full \
			eval.checkpoint_path=${CKPT_PATH} \
			trainer.devices=1; \
		done



train_promoter:
	python main.py \
		model=small \
		data=promoter \
		parameterization=new_diff \
		backbone=promoter \
		model.length=1024 \
		eval.compute_generative_perplexity=False \
		sampling.steps=100 \
		training.different_time=True \
		training.onehot_sparse=True \
		checkpointing.resume_from_ckpt=False \
		T=1000 \
		trainer.val_check_interval=100 \
		loader.global_batch_size=128 \
		trainer.devices=1; \

train_fb:
	python main.py \
		model=small \
		data=FB \
		parameterization=new_diff \
		backbone=FB \
		model.length=500 \
		eval.compute_generative_perplexity=False \
		sampling.steps=100 \
		training.different_time=True \
		training.onehot_sparse=True \
		checkpointing.resume_from_ckpt=False \
		T=1000 \
 		loader.global_batch_size=128 \
		trainer.val_check_interval=500 \
		gamma=0 \
		trainer.devices=1; \

sample_fb:
	$(eval CKPT_PATH:=/AIRvePFS/ai4science/users/yupei/test_slm_repo/ckpt/FB_best.ckpt)
	$(eval OUTPUT_DIR:=/AIRvePFS/ai4science/users/yupei/test_slm_repo/output/output_fb)
	python main.py \
		model=small \
		data=FB \
		parameterization=new_diff \
		backbone=FB \
		model.length=500 \
		eval.compute_generative_perplexity=False \
		sampling.steps=200 \
		sampling.outdir=${OUTPUT_DIR} \
		loader.global_batch_size=32 \
		training.different_time=True \
		training.onehot_sparse=True \
		mode=eval \
		eval.checkpoint_path=${CKPT_PATH} \
		gamma=2.7 \
		trainer.devices=1; \


train_mel:
	python main.py \
		model=small \
		data=Mel \
		parameterization=new_diff \
		backbone=Mel \
		model.length=500 \
		eval.compute_generative_perplexity=False \
		sampling.steps=100 \
		training.different_time=True \
		training.onehot_sparse=True \
		checkpointing.resume_from_ckpt=False \
		T=1000 \
 		loader.global_batch_size=128 \
		trainer.val_check_interval=100 \
		gamma=0 \
		trainer.devices=1; \

sample_mel:
	$(eval CKPT_PATH:=/AIRvePFS/ai4science/users/yupei/test_slm_repo/ckpt/Mel_best.ckpt)
	python main.py model=small \
		data=Mel \
		parameterization=new_diff \
		backbone=Mel \
		model.length=500 \
		eval.compute_generative_perplexity=False \
		sampling.steps=200 \
		training.different_time=True \
		training.onehot_sparse=True \
		mode=eval \
		eval.checkpoint_path=${CKPT_PATH} \
		gamma=3.3; \
