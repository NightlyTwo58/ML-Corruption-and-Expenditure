# P1
Names: Richard Cai (rjc432), Wenkai Zhao (wz459)

Abstract:

The effect of natural resource exports on the [HDI Index of](https://hdr.undp.org/data-center/human-development-index#/indicies/HDI) countries worldwide has been a historically contested topic, with some argueing that it accelerates local development through capital inflow, while others argue that it actually renders a net negative effect through fostering curruption & complacency through the creation of a rentier state. The negative impacts of such exports are often cited in phenomenon such as the supposed "[resource curse](https://wikipedia.org/wiki/resource_curse)" and "[dutch disease](https://wikipedia.org/wiki/dutch_disease)". We aim to investigate these two competiting but not necessarily exclusionary hypothesises through one unified index to acheieve a net determination on the true impacts of natural resource exports. We examined the impact of cereal, oil, lumber, rare earth metal, and ore exports from countries worldwide from the period 2010~2020 on their "[human development index (HDI)](https://hdr.undp.org/data-center/human-development-index#/indicies/HDI)", an composite index of life expectancy, education, and standard of living.

Our analysis primarily was composed of linear regression, K-means clustering, and some additional nonlinear regression. These were augmented by various measures of accuracy and visualizations to confirm trends, such as RMS and trendline visualizations.

Dataset Source:
1. [UN Comtrade](https://comtradeplus.un.org)
[Oil dataset](https://comtradeplus.un.org/TradeFlow?Frequency=A&Flows=X&CommodityCodes=2709&Partners=0&Reporters=all&period=all&AggregateBy=none&BreakdownMode=plus)
2. [HDI Data, UNDP](https://hdr.undp.org/sites/default/files/2023-24_HDR/HDR23-24_Composite_indices_complete_time_series.csv)

Research tool:
1.Google colab
2.python
3.Excel

Research process:
1. Organize data using Excel, sort and delete useless rows and columns.
2. Merge CVS tables with Python and reorganize data again.
3. Draw a scatter plot using Python and draw preliminary conclusions
4. Apply linear regression, cluster analysis, and nonlinear regression.
5. Draw the final conclusion.
