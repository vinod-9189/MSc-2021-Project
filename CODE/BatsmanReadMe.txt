
Required Package:
	pandas

Procedure:
	1. Import IPL Ball-by-Ball 2008-2020.csv
	
	2. Groupby Batsman, call compute(), for each grouped row via apply() function.
	
	3. Compute() function.
		
		a.) No of Matches Played:
			
			Id is a unique number assigned to each match. Find count of unique id’s for each batsman. 
		
		b.) No Of Not Outs:

			Compute no of times a batsman got out, from is_wicket column. No of matches played - No of outs 
		gives No of not outs.
		
		c.) Total Runs:

			To obtain total runs scored by a batsman, perform sum() operation on batsman_runs column. 
			
		d.) No Of Balls Faced:

			batsman_runs contain runs scored by batsman for each ball, by performing count() on this column we 
		would obtain balls faced by a batsman.
			
		e.) Highest Score:

			Groupby Id, then performing sum() function would yeild us the scores of batsman in each match. By applying
		max() function on list of scores highest score can be obtained.
			
		f.) Average Score:

			Total runs scored by batsman / No of Outs.
			
		g.) Strike Rate:
		
			Total runs scored by a batsman / total balls faced.
			
			Total balls faced faced can be computed by performing count() function on ball column.

		h.) 100's:

		    No of hundreds scored by a player.

		i.) 50's:

		    No of fifties scored by a player.

		j.) 6's:

		    No of Sixes hit by a batsmen.

		k.) 4's:

		    Nof fours hit by a batsmen.