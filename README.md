# P1
Names: Richard Cai (rjc432), Wenkai Zhao (wz459)

Abstract:

The effect of natural resource exports on the [HDI Index of](https://hdr.undp.org/data-center/human-development-index#/indicies/HDI) countries worldwide has been a historically contested topic, with some argueing that it accelerates local development through capital inflow, while others argue that it actually renders a net negative effect through fostering curruption & complacency through the creation of a rentier state. The negative impacts of such exports are often cited in phenomenon such as the supposed "[resource curse](https://wikipedia.org/wiki/resource_curse)" and "[dutch disease](https://wikipedia.org/wiki/dutch_disease)". We aim to investigate these two competiting but not necessarily exclusionary hypothesises through one unified index to acheieve a net determination on the true impacts of natural resource exports. We examined the impact of cereal, oil, lumber, rare earth metal, and ore exports from countries worldwide from the period 2010~2020 on their "[human development index (HDI)](https://hdr.undp.org/data-center/human-development-index#/indicies/HDI)", an composite index of life expectancy, education, and standard of living.

Our analysis primarily was composed of linear regression, K-means clustering, and nonlinear regression. These were augmented by R-squared measures of accuracy and visualizations to confirm trends.

1. Some prelimary visualizations are printed first: plots of worldwide HDI and a scatterplot of different resource types against HDI.
2. The linear regressions fits lines to each resource type on a single graph against HDI for all countries over a range of time.
3. Our K-means clustering is set to produce 8 clusters per resource type and convienently exports a total output of country-labled 35 .csv files.
4. We fitted the nonlinear regression using a power law for quadractic analysis and each calculates a R-squared value.

Dataset Source:
1. [UN Comtrade](https://comtradeplus.un.org)
[Oil dataset](https://comtradeplus.un.org/TradeFlow?Frequency=A&Flows=X&CommodityCodes=2709&Partners=0&Reporters=all&period=all&AggregateBy=none&BreakdownMode=plus)
2. [HDI Data, UNDP](https://hdr.undp.org/sites/default/files/2023-24_HDR/HDR23-24_Composite_indices_complete_time_series.csv)

In our research, we used Excel, Google Collab, and a Python IDE.

Research process:
1. Organize data using Excel, sort and delete useless rows and columns.
2. Merge CVS tables with Python and reorganize data again.
3. Draw a scatter plot using Python and draw preliminary conclusions
4. Apply linear regression, cluster analysis, and nonlinear regression.
5. Draw the final conclusion.
