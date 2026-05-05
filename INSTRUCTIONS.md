You are going to write a pythonic model that will use the PMA92C20 and PFA92C20 mortality rates within ./data. You will write a function that can calculate the present value of an annuity based on these mortality rates. The function will have the following inputs:
1. Gender
2. Age 
3. Discount Rate
4. Annual mortality improvement rate
5. Single or joint
6. Age difference of second life (optional)
7. Joint life annuity type first death or second death (optional)

The function should be able to accept vectors and produce a vectorised result if required. It should also be able to accept scalars.

Once this is complete, create a shiny app that uses the pv function to show the present value of a joint life annuity along the y axis and age along the x. There should be one coloured line for Joint Life First Death and another coloured line for Joint Life Second Death. Keep the axes anchored from 50 to 80 for age, and from 0 to 30 for present value. The plot should use seaborn, be a line plot, have 2 x 2 facets with each gender/gender2 combination, and a slider to control the age difference. 