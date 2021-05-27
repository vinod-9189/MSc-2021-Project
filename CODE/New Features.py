import pandas as pd

BBB = pd.read_csv("../datasets/IPL Ball-by-Ball 2008-2020.csv")
#BBB = BBB.iloc[2:,:]

No_Of_Matches = BBB["id"].unique().shape[0]
Match_Id = BBB["id"].unique()

NEW_DATA = pd.DataFrame()

No_Of_Runs_In_Last_Over = list()              # No of Runs in last over
No_Of_Wickets_In_Last_Over = list()             # No of Wickets in last over

No_Of_Dots_In_Last_Over = list()
No_Of_Singles_In_Last_Over = list()
No_Of_Doubles_In_Last_Over = list()
No_Of_Fours_In_Last_Over = list()
No_Of_Sixes_In_Last_Over = list()
Team_Score = list()
Strike_Rate = list()
Economy_Rate = list()
Dots,Singles,Doubles,Fours,Sixes = [0,0,0,0,0]

for i in range(No_Of_Matches):
    print(i+1)
    DATA = BBB[BBB["id"]==Match_Id[i]]          # Individual Match
    for j in range(1,3):
        
        TEMP = DATA[DATA["inning"] == j]            # 1st Innings in a Match
        No_Of_Rows = TEMP["inning"].count()         # No Of Balls in an Inning
        TEMP = TEMP.sort_values(['over', 'ball'], ascending=[True, True])  # Sorting rows by overs

        for k in range(No_Of_Rows):
            Dots,Singles,Doubles,Fours,Sixes = [0,0,0,0,0]
            Team_Score.append(TEMP.iloc[:k,9].sum())
            if k < 12:
                No_Of_Runs_In_Last_Over.append(TEMP.iloc[:k ,7].sum())
                No_Of_Wickets_In_Last_Over.append(TEMP.iloc[:k ,11].sum())

                for l in range(0,k):
                    if TEMP.iloc[l,7] == 0 and TEMP.iloc[l,11] != 1:
                        Dots = Dots + 1
                    elif TEMP.iloc[l,7] == 1:
                        Singles = Singles + 1
                    elif TEMP.iloc[l,7] == 2:
                        Doubles = Doubles + 1
                    elif TEMP.iloc[l,7] == 4:
                        Fours = Fours + 1
                    elif TEMP.iloc[l, 7] == 6:
                        Sixes = Sixes + 1
                No_Of_Dots_In_Last_Over.append(Dots)
                No_Of_Singles_In_Last_Over.append(Singles)
                No_Of_Doubles_In_Last_Over.append(Doubles)
                No_Of_Fours_In_Last_Over.append(Fours)
                No_Of_Sixes_In_Last_Over.append(Sixes)

            else:
                No_Of_Runs_In_Last_Over.append(TEMP.iloc[k-6:k ,7].sum())
                No_Of_Wickets_In_Last_Over.append(TEMP.iloc[k-6:k, 11].sum())

                for l in range(k-12,k):
                    if TEMP.iloc[l, 7] == 0 and TEMP.iloc[l, 11] != 1:
                        Dots = Dots + 1
                    elif TEMP.iloc[l, 7] == 1:
                        Singles = Singles + 1
                    elif TEMP.iloc[l, 7] == 2:
                        Doubles = Doubles + 1
                    elif TEMP.iloc[l, 7] == 4:
                        Fours = Fours + 1
                    elif TEMP.iloc[l, 7] == 6:
                        Sixes = Sixes + 1

                No_Of_Dots_In_Last_Over.append(Dots)
                No_Of_Singles_In_Last_Over.append(Singles)
                No_Of_Doubles_In_Last_Over.append(Doubles)
                No_Of_Fours_In_Last_Over.append(Fours)
                No_Of_Sixes_In_Last_Over.append(Sixes)


            sr = TEMP.iloc[:k+1,[2,3,4,7,6,9]]
            balls_faced = sr[sr["batsman"]==sr.iloc[k]["batsman"]]["batsman_runs"].count()
            Strike_Rate.append(round((sr[sr["batsman"]==sr.iloc[k]["batsman"]]["batsman_runs"].sum()/balls_faced)*100,2))
            balls_bowled = sr[sr["bowler"] == sr.iloc[k]["bowler"]]["total_runs"].count()
            Economy_Rate.append(round((sr[sr["bowler"] == sr.iloc[k]["bowler"]]["total_runs"].sum() / (balls_bowled/6)), 2))

        TEMP["Runs In Last 6 balls"] = No_Of_Runs_In_Last_Over
        TEMP["Fall Of Wickets In Last 6 balls"] = No_Of_Wickets_In_Last_Over
        TEMP["No_Of_Dots_In_Last_Over"] = No_Of_Dots_In_Last_Over
        TEMP["No_Of_Singles_In_Last_Over"] = No_Of_Singles_In_Last_Over
        TEMP["No_Of_Doubles_In_Last_Over"] = No_Of_Doubles_In_Last_Over
        TEMP["No_Of_Fours_In_Last_Over"] = No_Of_Fours_In_Last_Over
        TEMP["No_Of_Sixes_In_Last_Over"] = No_Of_Sixes_In_Last_Over
        TEMP["Team_Score"] = Team_Score
        TEMP["Strike_Rate"] = Strike_Rate
        TEMP["Economy_Rate"] = Economy_Rate

        No_Of_Runs_In_Last_Over.clear()
        No_Of_Wickets_In_Last_Over.clear()
        No_Of_Dots_In_Last_Over.clear()
        No_Of_Singles_In_Last_Over.clear()
        No_Of_Doubles_In_Last_Over.clear()
        No_Of_Fours_In_Last_Over.clear()
        No_Of_Sixes_In_Last_Over.clear()
        Team_Score.clear()
        Strike_Rate.clear()
        Economy_Rate.clear()

        NEW_DATA = pd.concat([NEW_DATA,TEMP.iloc[:, :]],ignore_index=True)

# Finding 1st Inning Average Sore W.R.T Every Team....

SCORES = pd.DataFrame()

for i in range(No_Of_Matches):
    D = NEW_DATA[NEW_DATA["id"] == Match_Id[i]]
    for j in range(1):
        TEMP = D[D["inning"] == j+1]
        SCORES = SCORES.append(TEMP.tail(1), ignore_index=True)
SCORES = SCORES[["batting_team","bowling_team","Team_Score"]]
SCORES = SCORES.groupby(["batting_team","bowling_team"]).apply(lambda x: x["Team_Score"].mean()).reset_index(name="1st_Inning_Score")
print(SCORES)

# Adding Pressure Factor

Pressure = list()

for i in range(No_Of_Matches):
    DATA = NEW_DATA[NEW_DATA["id"] == Match_Id[i]]

    TEMP_1 = DATA[DATA["inning"] == 1]  # 1st Innings in a Match
    No_Of_Rows_1 = TEMP_1["inning"].count()  # No Of Balls in 1st Inning

    TEMP_2 = DATA[DATA["inning"] == 2]  # 2nd Innings in a Match
    No_Of_Rows_2 = TEMP_2["inning"].count()  # No Of Balls in 2nd Inning

    Batting_Team = TEMP_1.iloc[0, 16]
    Bowling_Team = TEMP_1.iloc[0, 17]
    Target = SCORES[(SCORES["batting_team"] == Batting_Team) & (SCORES["bowling_team"] == Bowling_Team)].iloc[0, 2]

    for k in range(No_Of_Rows_1):
        Pressure.append(Target - TEMP_1.iloc[k, 25])

    Target = TEMP_1.tail(1).iloc[0]["Team_Score"]

    for k in range(No_Of_Rows_2):
        Pressure.append(Target - TEMP_2.iloc[k, 25])

NEW_DATA["Pressure"] = Pressure


print(NEW_DATA.columns)
NEW_DATA.to_csv("../datasets/MODIFIED_BBB.csv")

PP = NEW_DATA.loc[ (NEW_DATA["over"]<5) | (NEW_DATA["over"]>14) ]
NPP = NEW_DATA.loc[ (NEW_DATA["over"]>4) & (NEW_DATA["over"]<15) ]

PP.to_csv("../datasets/PP.csv")
NPP.to_csv("../datasets/NPP.csv")

#print(NEW_DATA.iloc[:30,[2,3,7,9,11,18,19,20,21,22,23,2,25]])
