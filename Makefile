submit_tests:
	sbatch jobs/test_run.sbatch

show_errors:
	for f in logs/transition_rate_study_model_training/*.e; do cat $$f; done;

check_errors_kl1:
	for f in logs/transition_rate_study_model_training/transition_rate_study_model_training_kl1/*.e; do if [ $$(cat $$f | wc -l) -eq 1 ]; then echo $$f; fi; done;

check_errors_kl2:
	for f in logs/transition_rate_study_model_training/transition_rate_study_model_training_kl2/*.e; do if [ $$(cat $$f | wc -l) -eq 1 ]; then echo $$f; fi; done;

check_errors_mcd1:
	for f in logs/transition_rate_study_model_training/transition_rate_study_model_training_mcd1/*.e; do if [ $$(cat $$f | wc -l) -eq 1 ]; then echo $$f; fi; done;

check_errors_mcd2:
	for f in logs/transition_rate_study_model_training/transition_rate_study_model_training_mcd2/*.e; do if [ $$(cat $$f | wc -l) -eq 1 ]; then echo $$f; fi; done;