from results.predictions import monte


####----Settings----####
country = "thailand"
station = "PHUK"

N= 50 #Number of samples in Monte Carlo
years_predict = 300 #Window of results - around 500 first then adjust for better visual
confidence_level = 95 #Choose confidence as percentage (0 < p < 100)
offset = 60  #Offset linear part of modelling function for better fit

####----Visuals----####
pred_pos = 4  #position above y=0


####----Run____####
monte(country, station, N, years_predict, confidence_level, offset, pred_pos)