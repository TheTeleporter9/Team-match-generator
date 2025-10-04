import random
import math
from collections import defaultdict
import numpy as np

class TeamScheduler:
    def __init__(self, num_teams, num_games, num_tables, rankings=None):
        self.X = num_teams
        self.N = num_games
        self.T = num_tables
        self.teams = list(range(1, num_teams + 1))
        
        # If rankings not provided, assume team number = ranking
        self.rankings = rankings if rankings else {team: team for team in self.teams}
        
        # Algorithm parameters
        self.k = 0.7  # Ranking tolerance
        self.max_ranking_diff = 3  # Maximum allowed ranking difference
        self.consecutive_penalty = 2.0  # Penalty for consecutive games
        
        # Tracking variables
        self.match_history = []
        self.team_match_counts = defaultdict(int)
        self.pair_match_counts = defaultdict(int)
        self.last_played = defaultdict(list)  # Track when teams last played
        
    def calculate_match_probability(self, team1, team2, game_round):
        """Calculate probability of match between team1 and team2"""
        rank_diff = abs(self.rankings[team1] - self.rankings[team2])
        
        # Base probability based on ranking difference
        base_prob = math.exp(-rank_diff / self.k)
        
        # Penalty for recent matches between same pair
        pair_key = tuple(sorted([team1, team2]))
        pair_penalty = math.exp(-self.pair_match_counts[pair_key])
        
        # Penalty for consecutive games
        consecutive_penalty = 1.0
        if self.last_played[team1] and game_round - self.last_played[team1][-1] <= 1:
            consecutive_penalty /= self.consecutive_penalty
        if self.last_played[team2] and game_round - self.last_played[team2][-1] <= 1:
            consecutive_penalty /= self.consecutive_penalty
            
        return base_prob * pair_penalty * consecutive_penalty
    
    def get_feasible_pairings(self, game_round, max_ranking_diff=None):
        """Get all feasible pairings for current round"""
        if max_ranking_diff is None:
            max_ranking_diff = self.max_ranking_diff
            
        feasible_pairs = []
        
        for i in range(len(self.teams)):
            for j in range(i + 1, len(self.teams)):
                team1, team2 = self.teams[i], self.teams[j]
                
                # Check ranking difference
                rank_diff = abs(self.rankings[team1] - self.rankings[team2])
                if rank_diff > max_ranking_diff:
                    continue
                    
                # Check if teams played recently
                team1_recent = (self.last_played[team1] and 
                               game_round - self.last_played[team1][-1] <= 1)
                team2_recent = (self.last_played[team2] and 
                               game_round - self.last_played[team2][-1] <= 1)
                
                # Allow some flexibility but penalize in probability calculation
                feasible_pairs.append((team1, team2))
                
        return feasible_pairs
    
    def select_pairings(self, feasible_pairs, game_round):
        """Select optimal pairings from feasible pairs"""
        if not feasible_pairs:
            return []
            
        # Calculate weights for each pair
        weights = []
        for team1, team2 in feasible_pairs:
            weight = self.calculate_match_probability(team1, team2, game_round)
            weights.append(weight)
            
        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            probabilities = [1/len(weights)] * len(weights)
        else:
            probabilities = [w/total_weight for w in weights]
            
        return feasible_pairs, probabilities
    
    def generate_schedule(self):
        """Generate complete schedule"""
        schedule = []
        used_teams_per_round = set()
        
        for game_round in range(1, self.N + 1):
            print(f"Generating round {game_round}/{self.N}...")
            
            round_matches = []
            used_teams_this_round = set()
            available_teams = set(self.teams)
            
            # Try to create T matches for this round
            for table in range(1, self.T + 1):
                if len(available_teams) < 2:
                    break
                    
                # Get feasible pairings with increasing ranking tolerance if needed
                max_diff = 1
                selected_pair = None
                
                while max_diff <= self.max_ranking_diff and selected_pair is None:
                    feasible_pairs = self.get_feasible_pairings(game_round, max_diff)
                    
                    # Filter to only available teams
                    feasible_pairs = [
                        (t1, t2) for t1, t2 in feasible_pairs 
                        if t1 in available_teams and t2 in available_teams
                    ]
                    
                    if feasible_pairs:
                        pairs, probs = self.select_pairings(feasible_pairs, game_round)
                        
                        # Use weighted random selection
                        if pairs:
                            selected_pair = random.choices(pairs, weights=probs, k=1)[0]
                    
                    max_diff += 1
                
                if selected_pair:
                    team1, team2 = selected_pair
                    round_matches.append((team1, team2))
                    used_teams_this_round.add(team1)
                    used_teams_this_round.add(team2)
                    available_teams.discard(team1)
                    available_teams.discard(team2)
                    
                    # Update tracking
                    self.pair_match_counts[tuple(sorted([team1, team2]))] += 1
                    self.last_played[team1].append(game_round)
                    self.last_played[team2].append(game_round)
            
            schedule.append(round_matches)
            self.match_history.append(round_matches)
            
        return schedule
    
    def print_schedule(self, schedule):
        """Print the generated schedule"""
        print("\n" + "="*50)
        print("TEAM SCHEDULING RESULTS")
        print("="*50)
        
        for round_num, round_matches in enumerate(schedule, 1):
            print(f"\nRound {round_num}:")
            for table_num, match in enumerate(round_matches, 1):
                team1, team2 = match
                rank_diff = abs(self.rankings[team1] - self.rankings[team2])
                print(f"  Table {table_num}: Team {team1} (Rank {self.rankings[team1]}) vs "
                      f"Team {team2} (Rank {self.rankings[team2]}) | Rank diff: {rank_diff}")
    
    def analyze_schedule(self, schedule):
        """Analyze the schedule quality"""
        print("\n" + "="*50)
        print("SCHEDULE ANALYSIS")
        print("="*50)
        
        # Count matches per team
        team_matches = defaultdict(int)
        team_opponents = defaultdict(list)
        
        for round_matches in schedule:
            for team1, team2 in round_matches:
                team_matches[team1] += 1
                team_matches[team2] += 1
                team_opponents[team1].append(team2)
                team_opponents[team2].append(team1)
        
        print(f"\nMatches per team:")
        for team in sorted(self.teams):
            print(f"  Team {team} (Rank {self.rankings[team]}): {team_matches[team]} matches")
        
        # Analyze ranking differences
        rank_differences = []
        for round_matches in schedule:
            for team1, team2 in round_matches:
                rank_diff = abs(self.rankings[team1] - self.rankings[team2])
                rank_differences.append(rank_diff)
        
        print(f"\nRanking difference analysis:")
        print(f"  Average rank difference: {np.mean(rank_differences):.2f}")
        print(f"  Max rank difference: {max(rank_differences)}")
        print(f"  Min rank difference: {min(rank_differences)}")
        
        # Check for consecutive games
        consecutive_games = 0
        for team in self.teams:
            games = sorted(self.last_played[team])
            for i in range(1, len(games)):
                if games[i] - games[i-1] == 1:
                    consecutive_games += 1
        
        print(f"  Teams playing consecutive games: {consecutive_games} occurrences")

def get_user_input():
    """Get all required inputs from the user"""
    print("="*60)
    print("TEAM SCHEDULING SYSTEM")
    print("="*60)
    
    # Get basic parameters
    while True:
        try:
            num_teams = int(input("\nEnter number of teams (X): "))
            num_games = int(input("Enter number of games (N): "))
            num_tables = int(input("Enter number of tables (T): "))
            
            if num_teams < 2:
                print("Error: Need at least 2 teams")
                continue
            if num_tables * 2 > num_teams:
                print("Error: Not enough teams for the number of tables")
                print(f"With {num_tables} tables, you need at least {num_tables * 2} teams")
                continue
            if num_games < 1:
                print("Error: Need at least 1 game")
                continue
                
            break
        except ValueError:
            print("Error: Please enter valid numbers")
    
    # Get ranking method
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
        # Use team numbers as rankings
        rankings = {team: team for team in range(1, num_teams + 1)}
        print("Using team numbers as rankings")
    else:
        # Get custom rankings
        print(f"\nEnter rankings for each team (1 = best, {num_teams} = worst)")
        print("Teams with similar rankings will play against each other")
        
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
        
        # Show ranking summary
        print("\nTeam rankings summary:")
        for team in sorted(rankings.keys()):
            print(f"  Team {team}: Rank {rankings[team]}")
    
    # Get algorithm parameters
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
    
    return num_teams, num_games, num_tables, rankings, k_value, penalty_value

def main():
    """Main function to run the scheduling system"""
    try:
        # Get user input
        num_teams, num_games, num_tables, rankings, k_value, penalty_value = get_user_input()
        
        # Create scheduler
        scheduler = TeamScheduler(num_teams, num_games, num_tables, rankings)
        scheduler.k = k_value
        scheduler.consecutive_penalty = penalty_value
        
        # Generate schedule
        print("\n" + "="*50)
        print("GENERATING SCHEDULE...")
        print("="*50)
        
        schedule = scheduler.generate_schedule()
        
        # Display results
        scheduler.print_schedule(schedule)
        scheduler.analyze_schedule(schedule)
        
        # Option to regenerate
        while True:
            regenerate = input("\nGenerate a different schedule with same parameters? (y/n): ").lower()
            if regenerate == 'y':
                print("\nGenerating new schedule...")
                scheduler = TeamScheduler(num_teams, num_games, num_tables, rankings)
                scheduler.k = k_value
                scheduler.consecutive_penalty = penalty_value
                schedule = scheduler.generate_schedule()
                scheduler.print_schedule(schedule)
                scheduler.analyze_schedule(schedule)
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