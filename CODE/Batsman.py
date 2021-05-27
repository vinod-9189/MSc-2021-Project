import pandas as pd
import numpy as np

def compute(x):
    l = list()

    # No of Matches Played

    NoOfMatchesPlayed = len(list(x["id"].unique()))
    l.append(NoOfMatchesPlayed)

    # No of Not Outs

    NoOfOuts = x["is_wicket"].sum()
    l.append(NoOfMatchesPlayed - NoOfOuts)

    # Total Runs

    l.append(x["batsman_runs"].sum())

    # No Of Balls Faced

    l.append(x["batsman_runs"].count())

    # Highest Score

    l.append(x.groupby(["id"])["batsman_runs"].sum().max())

    # Average Score

    if NoOfOuts == 0:
        l.append(x["batsman_runs"].sum())
    else:
        l.append(round(x["batsman_runs"].sum() / NoOfOuts, 2))

    # Strike Rate

    l.append(round((x['batsman_runs'].sum() / x['ball'].count()) * 100, 2))

    # 100's

    hundred = 0
    fifty = 0
    runs = x.groupby(["id"])["batsman_runs"].sum()
    for e in runs:
        if e > 99:
            hundred = hundred + 1
        elif e > 49:
            fifty = fifty + 1
    l.append(hundred)

    # 50's

    l.append(fifty)

    # 6's

    six = 0
    four = 0
    for b in x["batsman_runs"]:
        if b == 6:
            six = six + 1
        elif b == 4:
            four = four + 1
    l.append(six)

    # 4's

    l.append(four)
    return (pd.Series(l, index=["Matches Played", "Not Outs", "Runs", "Balls Faced", "Highest Score", "Average",
                                "Strike Rate","100's","50's","6's","4's"]))


DATA = pd.read_csv("../datasets/IPL Ball-by-Ball 2008-2020.csv")
print(DATA)

temp = DATA.groupby(["batsman"]).apply(compute).reset_index()
print(temp)

print(temp[temp["batsman"] == "V Kohli"])