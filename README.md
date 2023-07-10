# CTMC-Traffic-Assignment
Computer Systems Performance Evaluation (CSPE) course final project on Spring 2023.

# About Project
This project was developed to manage Traffic Assignment via Markov Chains (CTMC processes).
In this project we model a city with a road graph. There are also a list of cars that which to travel from their source to their destination. They wish to choode the shortest path, and we have control over setting the path costs.
Our goal is to manage the traffic by setting costs for the roads to optimize our utility fuction, which is basically about reducing the density of cars in roads overall. which could also be infered as load bablncing.

# Some results
For a given graph (you can find it's data in the codes), the average path cost overall is charted below. It can be seen that over time, not only the costs have significantly decreased, 
but also it has reached a point of convergance in the overall cost, meaning we could stop iterating there.

‌
   

![image](https://github.com/npourazin/CTMC-Traffic-Assignment/assets/44080169/8c8f96ae-a826-469b-ae34-3df55d64e496)

City Graph


‌


![image](https://github.com/npourazin/CTMC-Traffic-Assignment/assets/44080169/8e916feb-d3c4-4bea-867a-21cd500bbabb)

Average Path Cost over the iterations





# Sources
[1] E. Crisostomi, S. Kirkland, and R. Shorten, ‘A Google-like model of road network dynamics and its application to regulation and control’, International Journal of Control, vol. 84, no. 3, pp. 633–651, 2011.

[2] S. Salman and S. Alaswad, ‘Alleviating road network congestion: Traffic pattern optimization using Markov chain traffic assignment’, Computers & Operations Research, vol. 99, pp. 191–205, 2018. 

[3] D. Margalit and J. Rabinoff, Interactive linear algebra. 2019.
