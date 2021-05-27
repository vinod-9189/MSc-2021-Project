import pandas as pd
import numpy as np

BBB = pd.read_csv("datasets/IPL Ball-by-Ball 2008-2020.csv")
#print(BBB)

MATCHES = pd.read_csv("datasets/MATCHES.csv")

# BATSMAN ATTRIBUTES...

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


BATSMAN = BBB.groupby(["batsman"]).apply(compute).reset_index()
#print(BATSMAN)

print(BATSMAN[BATSMAN["batsman"] == "V Kohli"])
#BATSMAN.to_csv("datasets/BATSMAN.csv",index=False)

# BOWLER ATTRIBUTES...


def stats(data):
    # No of matches
    mats = len(data["id"].unique())

    # No of Innings

    innings = data.groupby(["bowler", "id", "inning"]).apply(lambda x: len(x["inning"].unique())).reset_index(
        name="inns")
    innings = innings.groupby("bowler").apply(lambda x: np.sum(x["inns"])).reset_index(name="inns")

    # No of Overs
    data6 = data[(data["extras_type"] != 'wides') & (data["extras_type"] != 'noballs') & (
            data["extras_type"] != 'penalty')]
    data6 = data6.groupby(["bowler", "id", "inning"]).apply(lambda x: len(x["ball"]) / 6).reset_index(name="overs")
    overs = data6.groupby("bowler").apply(lambda x: np.sum(x["overs"])).reset_index(name="overs")

    # No of Runs Conced
    #     The number of runs conceded by a bowler is determined as the total number of
    #     runs that the opposing side have scored while the bowler was bowling, excluding
    #     any byes, leg byes, or penalty runs.

    data2 = data[ (data["extras_type"] != 'byes') & (data["extras_type"] != 'legbyes') & (data["extras_type"] != 'penalty')]
    runs = np.sum(data2["total_runs"])

    # No of wickets
    data1 = data[data['dismissal_kind'] != 'run out']
    wkts = np.sum(data1["is_wicket"])

    # Average Rate
    if wkts != 0:
        avg = runs / wkts
    else:
        avg = runs

    # Economy Rate
    econ = runs / float(overs["overs"])

    # Strike Rate (Wicket Taking Ability)
    SR = (float(overs["overs"]) * 6) / (wkts + 0.0001)

    # No of 4wickets
    data4 = data1.groupby(["bowler", "id", "inning"]).apply(
        lambda x: 1 if (np.sum(x["is_wicket"]) == 4) else 0).reset_index(name="4wkts")
    no_of_4_wkts = data4.groupby("bowler").apply(lambda x: np.sum(x["4wkts"])).reset_index(name='4wkts')
    no_of_4_wkts = int(no_of_4_wkts["4wkts"])

    # No of 5wickets
    data4 = data1.groupby(["bowler", "id", "inning"]).apply(
        lambda x: 1 if (np.sum(x["is_wicket"]) == 5) else 0).reset_index(name="5wkts")
    no_of_5_wkts = data4.groupby("bowler").apply(lambda x: np.sum(x["5wkts"])).reset_index(name='5wkts')
    no_of_5_wkts = int(no_of_5_wkts["5wkts"])



    return (pd.Series(
        [mats, int(innings["inns"]), float(overs["overs"]), runs, wkts, round(avg, 2), round(econ, 2), round(SR, 2),
         no_of_4_wkts, no_of_5_wkts],
        index=['Mat', 'Inns', 'Overs', 'Runs', 'Wickets', 'Average', 'Economy', 'SR', '4Wkts', '5Wkts']))


bowlers_data = BBB.groupby("bowler").apply(stats).reset_index().sort_values(by="Wickets", ascending=False)
bowlers_data = pd.DataFrame(bowlers_data)
print(bowlers_data.head())
#bowlers_data.to_csv("datasets/BOWLER.csv",index=False)
# MATCHES ATTRIBUTES...

match=MATCHES.drop(['neutral_venue','eliminator','method','umpire1','umpire2','result_margin','team1','team2'],axis = 1)
print(match)

# DataFrame-1

#creating pitch

match.loc[(match['city'] == 'Bangalore') | (match['city'] == 'Bengaluru') | (match['city'] == 'Delhi')
          | (match['city'] == 'Kolkata') | (match['city'] == 'Mumbai') | (match['city'] == 'Nagpur')
          | (match['city'] == 'Dharamsala') | (match['city'] == 'Kochi') | (match['city'] == 'Johannesburg')
          ,'Pitch'] = 'Grass'
match.loc[(match['city'] == 'Chandigarh') | (match['city'] == 'Chennai') | (match['city'] == 'Visakhapatnam')
          | (match['city'] == 'Jaipur') | (match['city'] == 'Ahmedabad')| (match['city'] == 'Ranchi')
          | (match['city'] == 'Sharjah') | (match['city'] == 'Cape Town') | (match['city'] == 'Port Elizabeth')
          | (match['city'] == 'Centurion') , 'Pitch']  = 'Green'
match.loc[(match['city'] == 'Hyderabad') | (match['city'] == 'Dubai') | (match['city'] == 'Abu Dhabi')
          | (match['city'] == 'Pune') | (match['city'] == 'Cuttak')  | (match['city'] == 'Cuttack')
          | (match['city'] == 'Indore') | (match['city'] == 'Raipur') | (match['city'] == 'Rajkot')
          | (match['city'] == 'Kanpur') | (match['city'] == 'Kimberley') | (match['city'] == 'Bloemfontein')
          , 'Pitch'] = 'Dry and Dusty'
match.loc[(match['city'] == 'East London') , 'Pitch'] = 'Grass,Green'
match.loc[(match['city'] == 'Durban'), 'Pitch'] = 'Grass,Green,Dry and Dusty'

# match[match['Pitch'].isna()]

# print(match[1:10])

# team_scores

team_score = BBB.groupby(['id','inning'])['total_runs'].sum().unstack().reset_index()
team_score.columns = ['id','Team1_score', 'Team2_score']
matches_agg = pd.merge(match, team_score)
print(matches_agg)


# bowler

team_score1 = BBB[["bowler","id","over"]].groupby(['id','bowler']).count()/6
team_score1=team_score1.reset_index()
team_score1['over'] = team_score1['over'].astype(int)
# print(team_score1)


team_score2 = BBB[["bowler","id","is_wicket","total_runs"]].groupby(['id','bowler']).sum()
team_score2=team_score2.reset_index()
#team_score2['over'] = team_score1['over'].astype(int)
# print(team_score2)
ecom=pd.merge(team_score1,team_score2)
# print(ecom)

r=BBB[['id','bowler','total_runs','is_wicket','over']].groupby(['id','bowler'])
h=r.sum()
h=h.reset_index()
uid1=h['id'].unique()
size=len(uid1)
# print(size)


b=pd.DataFrame()
for i in range(size):
    t=h[h['id'] ==uid1[i]][['id','bowler','total_runs','is_wicket']]
    t=t[t['total_runs']==t['total_runs'].min()]
    t=t[t['is_wicket']==t['is_wicket'].max()]
    b=b.append(t,ignore_index=True)
# print(b)

fin=pd.merge(team_score1,b)
print(fin)

#batting
p=BBB[['id','batsman','batsman_runs']].groupby(['id','batsman'])
g=p.sum()
g=g.reset_index()
uid=g['id'].unique()
size=len(uid)
# print(size)

a=pd.DataFrame()
for i in range(size):
    t=g[g['id'] ==uid[i]][['id','batsman','batsman_runs']]
    t=t[t['batsman_runs']==t['batsman_runs'].max()]
    a=a.append(t,ignore_index=True)
# print(a)

match1=pd.merge(fin,a)
# print(match1)
final_data=pd.merge(matches_agg,match1)
print(final_data)

sort = final_data[['id','date','city','venue','Pitch','toss_decision','Team1_score','Team2_score','batsman','batsman_runs','bowler','over','total_runs','is_wicket','player_of_match','winner']]
print(sort)


# DataFrame-2

pol=MATCHES[['winner','toss_decision','venue']].groupby(['venue','winner']).toss_decision.value_counts().sort_index()
pol=pol.to_frame()
pol=pol.unstack().fillna(0).astype(int)
print(pol)

# TEAMS ATTRIBUTES...

DATA = MATCHES
print(DATA.head())
# name correction and replacing team names in short forms
DATA.replace('Rising Pune Supergiant', 'Rising Pune Supergiants', inplace=True)
DATA.replace('Delhi Daredevils', 'Delhi Capitals', inplace=True)
DATA.replace("Chennai Super Kings", "CSK", inplace=True)
DATA.replace("Deccan Chargers", "DC", inplace=True)
DATA.replace("Delhi Capitals", "DECAP", inplace=True)
DATA.replace("Gujarat Lions", "GL", inplace=True)
DATA.replace("Kings XI Punjab", "KXP", inplace=True)
DATA.replace("Kochi Tuskers Kerala", "KTK", inplace=True)
DATA.replace("Kolkata Knight Riders", "KKR", inplace=True)
DATA.replace("Mumbai Indians", "MI", inplace=True)
DATA.replace("Pune Warriors", "PW", inplace=True)
DATA.replace("Rajasthan Royals", "RR", inplace=True)
DATA.replace("Royal Challengers Bangalore", "RCB", inplace=True)
DATA.replace("Sunrisers Hyderabad", "SRH", inplace=True)
DATA.replace("Rising Pune Supergiants", "RPS", inplace=True)

a = DATA[['id']].drop_duplicates(subset=['id'])
a["batting_team"] = DATA.drop_duplicates(subset=['id'])[["team1"]]
a["bowling_team"] = DATA.drop_duplicates(subset=['id'])[["team2"]]

bat = DATA[['team1', 'id']].groupby(['team1']).count()
bat = bat.reset_index()

bow = DATA[['team2', 'id']].groupby(['team2']).count()
bow = bow.reset_index()

Team = pd.DataFrame()
Team['Team'] = bat['team1']
Team['1'] = bat['id']
Team['2'] = bow['id']
Team['Played'] = Team['1'] + Team['2']
Team = Team.drop(["1", "2"], axis=1)

# Won
mw = DATA[['result_margin', 'winner']].groupby('winner').count()
mw = mw.reset_index()

Team["won"] = mw["result_margin"]

# lost
Team['lost'] = (Team['Played'] - Team['won'])

# Toss_won

toss = DATA[['toss_winner', 'id']].groupby(['toss_winner']).count()
toss = toss.reset_index()

winner = DATA[DATA["toss_winner"] == DATA["winner"]]
winner = winner[["toss_winner", "winner"]]
winner = winner.groupby(["toss_winner"]).count().reset_index()

Team['Toss_Won'] = toss['id']

Team['TossWon&MatchWon'] = winner['winner']

# elect(bat or field) win count
bandf = DATA[["winner", "toss_decision"]].groupby('winner').toss_decision.value_counts().sort_index()
bandf = bandf.unstack().reset_index()

Team['ElectBatWin'] = bandf['bat'].fillna(0).astype(int)
Team['ElectFieldWin'] = bandf['field'].astype(int)

# seasonwin
win = DATA[["winner", "season"]].groupby(["season"]).tail(1).sort_values("season", ascending=True)
win = win.value_counts("winner").to_frame()
seasons = [3, 1, 0, 0, 2, 0, 0, 5, 0, 0, 0, 1, 1]
Team.insert(loc=8, column="season", value=seasons)

# print
print(Team)
