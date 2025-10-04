import random
import math
from collections import defaultdict
import numpy as np
import csv
import os
from datetime import datetime

class TeamScheduler:
    def __init__(self, num_teams, games_per_team, num_tables, rankings=None):
        self.X = num_teams
        self.games_per_team = games_per_team
        self.T = num_tables
        self.teams = list(range(1, num_teams + 1))
        
        # Calculate total games needed
        total_matches_needed = (num_teams * games_per_team) // 2
        self.N = math.ceil(total_matches_needed / num_tables)
        
        self.rankings = rankings if rankings else {team: team for team in self.teams}
        
        # Algorithm parameters
        self.k = 0.7
        self.max_ranking_diff = 3
        self.consecutive_penalty = 2.0
        
        # Tracking variables
        self.match_history = []
        self.team_match_counts = defaultdict(int)
        self.pair_match_counts = defaultdict(int)
        self.last_played = defaultdict(list)
        
    def calculate_match_probability(self, team1, team2, game_round):
        rank_diff = abs(self.rankings[team1] - self.rankings[team2])
        
        base_prob = math.exp(-rank_diff / self.k)
        
        pair_key = tuple(sorted([team1, team2]))
        pair_penalty = math.exp(-self.pair_match_counts[pair_key])
        
        consecutive_penalty = 1.0
        if self.last_played[team1] and game_round - self.last_played[team1][-1] <= 1:
            consecutive_penalty /= self.consecutive_penalty
        if self.last_played[team2] and game_round - self.last_played[team2][-1] <= 1:
            consecutive_penalty /= self.consecutive_penalty
            
        games_penalty = 1.0
        if self.team_match_counts[team1] >= self.games_per_team:
            games_penalty *= 0.1
        if self.team_match_counts[team2] >= self.games_per_team:
            games_penalty *= 0.1
            
        return base_prob * pair_penalty * consecutive_penalty * games_penalty
    
    def get_feasible_pairings(self, game_round, max_ranking_diff=None):
        if max_ranking_diff is None:
            max_ranking_diff = self.max_ranking_diff
            
        feasible_pairs = []
        
        for i in range(len(self.teams)):
            for j in range(i + 1, len(self.teams)):
                team1, team2 = self.teams[i], self.teams[j]
                
                if (self.team_match_counts[team1] >= self.games_per_team or 
                    self.team_match_counts[team2] >= self.games_per_team):
                    continue
                
                rank_diff = abs(self.rankings[team1] - self.rankings[team2])
                if rank_diff > max_ranking_diff:
                    continue
                    
                feasible_pairs.append((team1, team2))
                
        return feasible_pairs
    
    def select_pairings(self, feasible_pairs, game_round):
        if not feasible_pairs:
            return []
            
        weights = []
        for team1, team2 in feasible_pairs:
            weight = self.calculate_match_probability(team1, team2, game_round)
            weights.append(weight)
            
        total_weight = sum(weights)
        if total_weight == 0:
            probabilities = [1/len(weights)] * len(weights)
        else:
            probabilities = [w/total_weight for w in weights]
            
        return feasible_pairs, probabilities
    
    def generate_schedule(self):
        schedule = []
        
        for game_round in range(1, self.N + 1):
            print(f"Generating round {game_round}/{self.N}...")
            
            round_matches = []
            available_teams = set(self.teams)
            
            available_teams = {team for team in available_teams 
                             if self.team_match_counts[team] < self.games_per_team}
            
            for table in range(1, self.T + 1):
                if len(available_teams) < 2:
                    break
                    
                max_diff = 1
                selected_pair = None
                
                while max_diff <= self.max_ranking_diff and selected_pair is None:
                    feasible_pairs = self.get_feasible_pairings(game_round, max_diff)
                    
                    feasible_pairs = [
                        (t1, t2) for t1, t2 in feasible_pairs 
                        if t1 in available_teams and t2 in available_teams
                    ]
                    
                    if feasible_pairs:
                        pairs, probs = self.select_pairings(feasible_pairs, game_round)
                        
                        if pairs:
                            selected_pair = random.choices(pairs, weights=probs, k=1)[0]
                    
                    max_diff += 1
                
                if selected_pair:
                    team1, team2 = selected_pair
                    round_matches.append((team1, team2))
                    available_teams.discard(team1)
                    available_teams.discard(team2)
                    
                    self.pair_match_counts[tuple(sorted([team1, team2]))] += 1
                    self.team_match_counts[team1] += 1
                    self.team_match_counts[team2] += 1
                    self.last_played[team1].append(game_round)
                    self.last_played[team2].append(game_round)
            
            schedule.append(round_matches)
            self.match_history.append(round_matches)
            
            all_teams_done = all(self.team_match_counts[team] >= self.games_per_team 
                               for team in self.teams)
            if all_teams_done:
                print(f"All teams have played their required games. Stopping at round {game_round}.")
                break
            
        return schedule
    
    def print_schedule(self, schedule):
        print("\n" + "="*50)
        print("TEAM SCHEDULING RESULTS")
        print("="*50)
        print(f"Configuration: {self.X} teams, {self.games_per_team} games per team, {self.T} tables")
        print(f"Total rounds generated: {len(schedule)}")
        
        for round_num, round_matches in enumerate(schedule, 1):
            print(f"\nRound {round_num}:")
            for table_num, match in enumerate(round_matches, 1):
                team1, team2 = match
                rank_diff = abs(self.rankings[team1] - self.rankings[team2])
                print(f"  Table {table_num}: Team {team1} vs Team {team2} | Rank diff: {rank_diff}")
    
    def analyze_schedule(self, schedule):
        print("\n" + "="*50)
        print("SCHEDULE ANALYSIS")
        print("="*50)
        
        team_matches = defaultdict(int)
        team_opponents = defaultdict(list)
        
        for round_matches in schedule:
            for team1, team2 in round_matches:
                team_matches[team1] += 1
                team_matches[team2] += 1
                team_opponents[team1].append(team2)
                team_opponents[team2].append(team1)
        
        print(f"\nMatches per team (target: {self.games_per_team}):")
        for team in sorted(self.teams):
            status = "✓" if team_matches[team] == self.games_per_team else "✗"
            print(f"  Team {team}: {team_matches[team]} matches {status}")
        
        rank_differences = []
        for round_matches in schedule:
            for team1, team2 in round_matches:
                rank_diff = abs(self.rankings[team1] - self.rankings[team2])
                rank_differences.append(rank_diff)
        
        print(f"\nRanking difference analysis:")
        print(f"  Average rank difference: {np.mean(rank_differences):.2f}")
        print(f"  Max rank difference: {max(rank_differences)}")
        print(f"  Min rank difference: {min(rank_differences)}")
        
        consecutive_games = 0
        for team in self.teams:
            games = sorted(self.last_played[team])
            for i in range(1, len(games)):
                if games[i] - games[i-1] == 1:
                    consecutive_games += 1
                    print(f"  Team {team} played consecutive games in rounds {games[i-1]} and {games[i]}")
        
        print(f"\nTotal consecutive game occurrences: {consecutive_games}")
    
    def export_to_csv(self, schedule, filename=None):
        """Export schedule to CSV file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"team_schedule_{timestamp}.csv"
        
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write header
                writer.writerow(['Round', 'Table', 'Team 1', 'Team 2', 'Rank Team 1', 'Rank Team 2', 'Rank Difference'])
                
                # Write schedule data
                for round_num, round_matches in enumerate(schedule, 1):
                    for table_num, match in enumerate(round_matches, 1):
                        team1, team2 = match
                        rank1 = self.rankings[team1]
                        rank2 = self.rankings[team2]
                        rank_diff = abs(rank1 - rank2)
                        
                        writer.writerow([round_num, table_num, f'Team {team1}', f'Team {team2}', rank1, rank2, rank_diff])
                
                # Write summary section
                writer.writerow([])
                writer.writerow(['SCHEDULE SUMMARY'])
                writer.writerow(['Total Teams:', self.X])
                writer.writerow(['Games per Team:', self.games_per_team])
                writer.writerow(['Tables:', self.T])
                writer.writerow(['Total Rounds:', len(schedule)])
                writer.writerow(['Generated on:', datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
                
                # Write team matches summary
                writer.writerow([])
                writer.writerow(['TEAM MATCHES SUMMARY'])
                writer.writerow(['Team', 'Matches Played', 'Target'])
                
                team_matches = defaultdict(int)
                for round_matches in schedule:
                    for team1, team2 in round_matches:
                        team_matches[team1] += 1
                        team_matches[team2] += 1
                
                for team in sorted(self.teams):
                    writer.writerow([f'Team {team}', team_matches[team], self.games_per_team])
            
            print(f"\n✓ Schedule successfully exported to: {filename}")
            print(f"✓ File location: {os.path.abspath(filename)}")
            return filename
            
        except Exception as e:
            print(f"\n✗ Error exporting to CSV: {e}")
            return None

def get_user_input():
    print("="*60)
    print("TEAM SCHEDULING SYSTEM")
    print("="*60)
    
    while True:
        try:
            num_teams = int(input("\nEnter number of teams (X): "))
            games_per_team = int(input("Enter number of games PER TEAM: "))
            num_tables = int(input("Enter number of tables (T): "))
            
            if num_teams < 2:
                print("Error: Need at least 2 teams")
                continue
            if num_tables * 2 > num_teams:
                print("Error: Not enough teams for the number of tables")
                print(f"With {num_tables} tables, you need at least {num_tables * 2} teams")
                continue
            if games_per_team < 1:
                print("Error: Each team should play at least 1 game")
                continue
                
            total_matches = (num_teams * games_per_team) // 2
            min_rounds = math.ceil(total_matches / num_tables)
            print(f"Note: This will require approximately {min_rounds} rounds")
                
            break
        except ValueError:
            print("Error: Please enter valid numbers")
    
    print("\nRanking options:")
    print("1. Use team numbers as rankings (Team 1 = Rank 1, Team 2 = Rank 2, etc.)")
    print("2. Enter custom rankings")
    
    while True:
        try:
            ranking_choice = int(input("Choose ranking method (1 or 2): "))
            if ranking_choice in [1, 2]:
                break
            else:
                print("Please enter 1 or 2")
        except ValueError:
            print("Please enter 1 or 2")
    
    rankings = {}
    if ranking_choice == 1:
        rankings = {team: team for team in range(1, num_teams + 1)}
        print("Using team numbers as rankings")
    else:
        print(f"\nEnter rankings for each team (1 = best, {num_teams} = worst)")
        
        for team in range(1, num_teams + 1):
            while True:
                try:
                    rank = int(input(f"Enter ranking for Team {team}: "))
                    if 1 <= rank <= num_teams:
                        rankings[team] = rank
                        break
                    else:
                        print(f"Ranking must be between 1 and {num_teams}")
                except ValueError:
                    print("Please enter a valid number")
        
        print("\nTeam rankings summary:")
        for team in sorted(rankings.keys()):
            print(f"  Team {team}: Rank {rankings[team]}")
    
    print("\nAlgorithm Parameters (press Enter for default values):")
    
    try:
        k_input = input(f"Ranking tolerance [default: 0.7] (lower = stricter): ")
        k_value = float(k_input) if k_input.strip() else 0.7
    except ValueError:
        k_value = 0.7
        print("Using default value 0.7")
    
    try:
        penalty_input = input(f"Consecutive game penalty [default: 2.0] (higher = stricter): ")
        penalty_value = float(penalty_input) if penalty_input.strip() else 2.0
    except ValueError:
        penalty_value = 2.0
        print("Using default value 2.0")
    
    return num_teams, games_per_team, num_tables, rankings, k_value, penalty_value

def main():
    try:
        num_teams, games_per_team, num_tables, rankings, k_value, penalty_value = get_user_input()
        
        scheduler = TeamScheduler(num_teams, games_per_team, num_tables, rankings)
        scheduler.k = k_value
        scheduler.consecutive_penalty = penalty_value
        
        print("\n" + "="*50)
        print("GENERATING SCHEDULE...")
        print("="*50)
        
        schedule = scheduler.generate_schedule()
        
        scheduler.print_schedule(schedule)
        scheduler.analyze_schedule(schedule)
        
        # Export to CSV
        csv_file = scheduler.export_to_csv(schedule)
        
        # Option to regenerate
        while True:
            regenerate = input("\nGenerate a different schedule with same parameters? (y/n): ").lower()
            if regenerate == 'y':
                print("\nGenerating new schedule...")
                scheduler = TeamScheduler(num_teams, games_per_team, num_tables, rankings)
                scheduler.k = k_value
                scheduler.consecutive_penalty = penalty_value
                schedule = scheduler.generate_schedule()
                scheduler.print_schedule(schedule)
                scheduler.analyze_schedule(schedule)
                csv_file = scheduler.export_to_csv(schedule)  # Export the new schedule too
            elif regenerate == 'n':
                break
            else:
                print("Please enter 'y' or 'n'")
                
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user")
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()