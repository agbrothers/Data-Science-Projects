import pandas as pd
import requests


""" Assembling DF's of stats for players ranked top 50 in launch speed for 2015-2019 """

headers = ["batter", "year", "avg_launch_speed", "avg_height", "avg_distance", "avg_launch_angle"]
all_dfs = dict()  # Creating the dictionary that will hold the DF for each year (2015-2019)

for year in range(2015, 2020):
    url = "https://lookup-service-prod.mlb.com/json/named.psc_leader_hit_avg.bam?season={}&".format(year)
    page = requests.get(url)
    rawd = page.json()
    hitters = rawd["psc_leader_hit_avg"]["queryResults"]["row"]
    final_df = pd.DataFrame(columns=headers)  # Creating a year's DF, to be filled in with for loop below

    for i in range(50):  # Filling DF with top 50 players' data
        row = []
        for stat in headers:              # Filling in a row with a player's stats
            row.append(hitters[i][stat])
        tempdf = pd.DataFrame(data=row).transpose()
        tempdf.columns = headers
        final_df = pd.concat([final_df, tempdf], ignore_index=True)  # Adding a player's stats as a new row in our DF
    all_dfs["df_{}".format(year)] = final_df  # Adding a year's DF to our dictionary containing all our DF's
print(all_dfs)


""" DF of yearly averages for players ranked top 50 in launch speed """

Year = range(2015, 2020)
Launch_Speed = []  # Creating columns of DF, to be filled in with for loop below
Height = []        #
Distance = []      #
Launch_Angle = []  #

for year in all_dfs.keys():  # Assigning the average of each stat for each year to the objects above
    Launch_Speed.append(pd.to_numeric(all_dfs[year]["avg_launch_speed"]).mean())
    Height.append(pd.to_numeric(all_dfs[year]["avg_height"]).mean())
    Distance.append(pd.to_numeric(all_dfs[year]["avg_distance"]).mean())
    Launch_Angle.append(pd.to_numeric(all_dfs[year]["avg_launch_angle"]).mean())

avgs_by_year = {"Year": Year,
                "Launch Speed": Launch_Speed,
                "Height": Height,
                "Distance": Distance,
                "Launch Angle": Launch_Angle}

avgs_by_year = pd.DataFrame(avgs_by_year)
avgs_by_year.to_csv("DSU_juice.csv")
