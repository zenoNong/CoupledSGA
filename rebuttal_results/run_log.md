# Rebuttal Evaluation Run Log

## Checkpoints Used
* **Model A (Baseline):** `d:\CoupledSGA\checkpoints\ode_predcls_future_5\ode_predcls_future_5_epoch_3.tar`
* **Model B (CoupledSGA λ=0):** `d:\CoupledSGA\checkpoints\model_b_nosc\model_b_nosc_epoch_3.tar`
* **Model C (CoupledSGA λ=0.1, estart=2):** `d:\CoupledSGA\checkpoints\model_c_sc\model_c_sc_epoch_3.tar`

## Reproduction Check (With-Constraint R@20)
* **Model A:** 44.15
* **Model B:** 42.21
* **Model C:** 35.18
*(Note: These match the retrained weights provided in progress.txt)*

## Command Lines Executed
```powershell
$env:PYTHONPATH = "d:\CoupledSGA"

# Evaluate Model A
& d:\CoupledSGA\venv\Scripts\python.exe d:\CoupledSGA\test_sga_methods.py --method_name ode --datasize mini --data_path "d:\CoupledSGA\datasets\ag" --save_path "d:\CoupledSGA\checkpoints" --ckpt "d:\CoupledSGA\checkpoints\ode_predcls_future_5\ode_predcls_future_5_epoch_3.tar"

# Evaluate Model B
& d:\CoupledSGA\venv\Scripts\python.exe d:\CoupledSGA\test_sga_methods.py --method_name coupledsga --lambda_sc 0.0 --datasize mini --data_path "d:\CoupledSGA\datasets\ag" --save_path "d:\CoupledSGA\checkpoints" --ckpt "d:\CoupledSGA\checkpoints\model_b_nosc\model_b_nosc_epoch_3.tar"

# Evaluate Model C
& d:\CoupledSGA\venv\Scripts\python.exe d:\CoupledSGA\test_sga_methods.py --method_name coupledsga --lambda_sc 0.1 --datasize mini --data_path "d:\CoupledSGA\datasets\ag" --save_path "d:\CoupledSGA\checkpoints" --ckpt "d:\CoupledSGA\checkpoints\model_c_sc\model_c_sc_epoch_3.tar"

# Compute Metrics
& d:\CoupledSGA\venv\Scripts\python.exe d:\CoupledSGA\structural_metrics.py
```
