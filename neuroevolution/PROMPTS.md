1) Implement the described experiment 1. 2) Fill the missing `Plots` section. It should describe the plots that you will generate as a result of the experiments. The plots need to cover all experiments. 3) Run the experiment.

Do not look for files outside of this folder.

modify @experiment_1.py such that there is no max num_epochs or num_generations. Have some logic to check for convergence and kill the experiment when that is reached. Also save the experiment results in files so that we can easily replot. Also please have the plot update itself in real time from the beginning of the experimentation. Also instead of looping over all optimization run types, just have it run 1 at a time and run the script as many times as necessary. And make it so that every time I re-run the script, no matter what I change, the plot updates itself in real-time. neuroevolution methods can look to be stalling for a while, so be leniant in your convergence detection

---

Few things:
1. I got this error on a run after starting another run in parallel
```
 DL Epoch 310: Loss=0.1058, F1=0.9138
  DL Epoch 311: Loss=0.1054, F1=0.9136
  DL Epoch 312: Loss=0.1050, F1=0.9138
Traceback (most recent call last):
  File "C:\Users\Max\Dropbox\repos\maximilienleclei\behaviour_imitation_research\projects\scaling_bc_to_perfection\experiment_1.py", line 775, in <module>
    main()
  File "C:\Users\Max\Dropbox\repos\maximilienleclei\behaviour_imitation_research\projects\scaling_bc_to_perfection\experiment_1.py", line 759, in main
    run_single_method(
  File "C:\Users\Max\Dropbox\repos\maximilienleclei\behaviour_imitation_research\projects\scaling_bc_to_perfection\experiment_1.py", line 672, in run_single_method
    train_deep_learning(
  File "C:\Users\Max\Dropbox\repos\maximilienleclei\behaviour_imitation_research\projects\scaling_bc_to_perfection\experiment_1.py", line 399, in train_deep_learning
    update_plot(dataset_name)
  File "C:\Users\Max\Dropbox\repos\maximilienleclei\behaviour_imitation_research\projects\scaling_bc_to_perfection\experiment_1.py", line 99, in update_plot
    results: dict[str, dict] = load_all_results(dataset_name)
                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Max\Dropbox\repos\maximilienleclei\behaviour_imitation_research\projects\scaling_bc_to_perfection\experiment_1.py", line 90, in load_all_results
    results[method_name] = json.load(f)
                           ^^^^^^^^^^^^
  File "C:\Users\Max\AppData\Local\Programs\Python\Python312\Lib\json\__init__.py", line 293, in load
    return loads(fp.read(),
           ^^^^^^^^^^^^^^^^
  File "C:\Users\Max\AppData\Local\Programs\Python\Python312\Lib\json\__init__.py", line 346, in loads
    return _default_decoder.decode(s)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Max\AppData\Local\Programs\Python\Python312\Lib\json\decoder.py", line 338, in decode
    obj, end = self.raw_decode(s, idx=_w(s, 0).end())
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Max\AppData\Local\Programs\Python\Python312\Lib\json\decoder.py", line 356, in raw_decode
    raise JSONDecodeError("Expecting value", s, err.value) from None
json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
```
2. I don't like the convergence speed analysis graph, remove it
3. On the training loss/fitness curve plot, do not display the methods optimizing F1. Only keep methods optimizing cross entropy. I see that you made neuroevolution maximize negative cross entropy, can you make it minimize positive cross entropy instead (which I believe is what you made DL do?)
4. For the training loss/fitness curve and test macro f1 score curve plots, replace the x axis from iteration/generation to `Runtime %`. This entails that all methods will span from left to right.
5. Periodically save checkpoints for any optimization method ran, so I can resume if any crashes.