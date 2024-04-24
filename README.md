Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
# WoCC_public
Proposed algorithms for aggregating crowds' opinions in different domains: Sports, Politics, Economics, and Climate.

The data (which is not added here) comes from a project called [Wisdom of the Crowd: Crowd Analysis Project](https://woccap.com/). In the prediction survey, participants have to answer many questionnaires including their demographics and their cognitive abilities. For each domain, they need to predict one thing, such as the price of oil in the next month or number of days with a warmer than average temperature in a location. Their prediction for the two previous months was also available. There was a competition between the research teams to propose an algorithm which give the closest prediction to the true value, based on the prediction survey.

In our team, [Tom](https://tomstafford.sites.sheffield.ac.uk/) and I have proposed multiple algorithm for each domain. First we proposed excluding people who did not pay enough attention to other questionnaires. These exclusions were quite liberal. After that, for the Climate domain, we used simple averaging. However for other domains, since the competition only allowed for one algorithm per domain and we did not have access to the true value, we proposed applying the algorithm to the previous month, estimate the deviation from true value and compare the algorithms. In the end, the algorithm with lowest deviation was chosen to use for the actual prediction.
Our proposed algorithms, included median, identifying top predictors and only using their prediction (top_k_mean), contribution weighted algorithm (CWM) which identifies low-performing individuals and removes them, and diversifying responses where predictions are picked based on increasing the standard deviation (diversify_responses). Find more explanations and possible references inside codes.
