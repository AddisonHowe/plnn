submit_tests:
	sbatch jobs/test_run.sbatch

show_errors:
	for f in logs/transition_rate_study_model_training/*.e; do cat $$f; done;

check_errors:
	for f in logs/transition_rate_study_model_training/*.e; do if [ $$(cat $$f | wc -l) -gt 1 ]; then echo $$f; fi; done;