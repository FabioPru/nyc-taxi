# nyc-taxi
Final project for CS182. Contains the following

* **src/** Python source code for classes, models, etc.
* **data/** data about NYC taxis downloaded from https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page.
* **explain.ipynb** explanation about MDP model we chose, graphs, states, actions, policies.
* **baseline.ipynb** evaluating some baseline policies.
* **policy_improvement.ipynb** implementing an approximate form of policy improvement. Fit an estimator (NN or linear model) to the rewards of sample episodes, use that to perform approximate policy evaluation.