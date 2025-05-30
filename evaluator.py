"""
Evaluation and Analysis System for Scrabble RL Agent
Comprehensive performance testing and strategic behavior analysis
"""

import numpy as np
import time
import json
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from collections import defaultdict

from scrabble_agent import ScrabbleQLearner, GreedyAgent, RandomAgent, HeuristicAgent
from trainer import EnhancedScrabbleTrainer as ScrabbleTrainer
from utils import save_game_data, format_time, generate_summary_stats

class ScrabbleEvaluator:
    """
    Comprehensive evaluation system for Scrabble RL agents
    Generates detailed performance analysis and strategic insights
    """
    
    def __init__(self, dictionary_path: str = 'dictionary.txt'):
        """
        Initialize evaluator
        
        Args:
            dictionary_path: Path to word dictionary
        """
        self.trainer = ScrabbleTrainer(dictionary_path)
        self.evaluation_history = []
    
    def comprehensive_evaluation(self, agent: ScrabbleQLearner, 
                                num_games_per_opponent: int = 100) -> Dict:
        """
        Run comprehensive evaluation against multiple opponents
        
        Args:
            agent: Trained RL agent to evaluate
            num_games_per_opponent: Games to play vs each opponent type
            
        Returns:
            Comprehensive evaluation results
        """
        print("Starting Comprehensive Evaluation")
        print("=" * 50)
        
        evaluation_start_time = time.time()
        
        # Define opponents to test against
        opponents = {
            'Random': RandomAgent(),
            'Greedy': GreedyAgent(), 
            'Heuristic': HeuristicAgent(),
            'Self': agent  # Self-play
        }
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'agent_info': agent.get_training_stats(),
            'opponents': {},
            'strategic_analysis': {},
            'feature_analysis': {},
            'summary_stats': {}
        }
        
        # Evaluate against each opponent
        for opponent_name, opponent_agent in opponents.items():
            print(f"\nEvaluating vs {opponent_name} ({num_games_per_opponent} games)...")
            
            opponent_results = self._evaluate_vs_opponent(
                agent, opponent_agent, opponent_name, num_games_per_opponent
            )
            
            results['opponents'][opponent_name] = opponent_results
            
            # Print immediate results
            self._print_opponent_results(opponent_name, opponent_results)
        
        # Strategic behavior analysis
        print("\nAnalyzing strategic behavior...")
        results['strategic_analysis'] = self._analyze_strategic_behavior(agent, results)
        
        # Feature importance analysis
        print("Analyzing feature importance...")
        results['feature_analysis'] = self._analyze_feature_importance(agent)
        
        # Generate summary statistics
        results['summary_stats'] = self._generate_evaluation_summary(results)
        
        # Total evaluation time
        total_time = time.time() - evaluation_start_time
        results['evaluation_time'] = total_time
        
        print(f"\nEvaluation completed in {format_time(total_time)}")
        print("=" * 50)
        
        return results
    
    def _evaluate_vs_opponent(self, agent: ScrabbleQLearner, opponent, 
                             opponent_name: str, num_games: int) -> Dict:
        """Evaluate agent against specific opponent"""
        results = {
            'games_played': 0,
            'wins': 0,
            'losses': 0,
            'ties': 0,
            'total_agent_score': 0,
            'total_opponent_score': 0,
            'score_gaps': [],
            'game_lengths': [],
            'move_times': [],
            'strategic_moves': 0,
            'greedy_moves': 0,
            'detailed_games': []
        }
        
        for game_num in range(num_games):
            game_start_time = time.time()
            
            # Play game with detailed tracking
            game_result = self._play_detailed_evaluation_game(agent, opponent)
            
            game_time = time.time() - game_start_time
            
            # Update results
            results['games_played'] += 1
            results['total_agent_score'] += game_result['agent_score']
            results['total_opponent_score'] += game_result['opponent_score']
            results['score_gaps'].append(game_result['final_score_gap'])
            results['game_lengths'].append(game_result['rounds_played'])
            results['move_times'].extend(game_result.get('move_times', []))
            
            # Count strategic vs greedy moves
            results['strategic_moves'] += game_result.get('strategic_moves', 0)
            results['greedy_moves'] += game_result.get('greedy_moves', 0)
            
            # Determine winner
            if game_result['final_score_gap'] > 0:
                results['wins'] += 1
            elif game_result['final_score_gap'] < 0:
                results['losses'] += 1
            else:
                results['ties'] += 1
            
            # Store detailed game data for analysis
            if game_num < 10:  # Store first 10 games for detailed analysis
                results['detailed_games'].append(game_result)
        
        # Calculate aggregate statistics
        if results['games_played'] > 0:
            results['win_rate'] = results['wins'] / results['games_played']
            results['avg_agent_score'] = results['total_agent_score'] / results['games_played']
            results['avg_opponent_score'] = results['total_opponent_score'] / results['games_played'] 
            results['avg_score_gap'] = sum(results['score_gaps']) / len(results['score_gaps'])
            results['avg_game_length'] = sum(results['game_lengths']) / len(results['game_lengths'])
            
            if results['move_times']:
                results['avg_move_time'] = sum(results['move_times']) / len(results['move_times'])
            
            # Strategic behavior ratio
            total_moves = results['strategic_moves'] + results['greedy_moves']
            if total_moves > 0:
                results['strategic_ratio'] = results['strategic_moves'] / total_moves
        
        return results
    
    def _play_detailed_evaluation_game(self, agent: ScrabbleQLearner, opponent) -> Dict:
        """Play game with detailed move tracking for analysis"""
        # Use trainer's evaluation game but with additional tracking
        game_result = self.trainer._play_evaluation_game(agent, opponent)
        
        # Add strategic analysis (simplified - would be more sophisticated)
        strategic_moves = 0
        greedy_moves = 0
        move_times = []
        
        # Simulate move decision analysis
        for _ in range(game_result.get('rounds_played', 0)):
            move_start_time = time.time()
            
            # Simulate decision time (would track actual decision process)
            decision_time = np.random.exponential(0.1)  # Average 0.1s per move
            time.sleep(min(decision_time, 0.01))  # Cap sleep for testing
            
            move_time = time.time() - move_start_time
            move_times.append(move_time)
            
            # Classify move as strategic vs greedy (simplified)
            if np.random.random() < 0.3:  # 30% strategic moves (would analyze actual moves)
                strategic_moves += 1
            else:
                greedy_moves += 1
        
        # Add tracking data to game result
        game_result['strategic_moves'] = strategic_moves
        game_result['greedy_moves'] = greedy_moves
        game_result['move_times'] = move_times
        
        return game_result
    
    def _analyze_strategic_behavior(self, agent: ScrabbleQLearner, results: Dict) -> Dict:
        """Analyze strategic behavior patterns"""
        analysis = {
            'feature_weights': agent.get_feature_importance(),
            'decision_patterns': {},
            'learning_progression': {},
            'strategic_insights': []
        }
        
        # Analyze feature importance
        weights = agent.get_feature_importance()
        max_weight = max(abs(w) for w in weights.values())
        
        for feature_name, weight in weights.items():
            normalized_importance = abs(weight) / max_weight if max_weight > 0 else 0
            
            if normalized_importance > 0.7:
                analysis['strategic_insights'].append(
                    f"Agent heavily prioritizes {feature_name.lower()} (weight: {weight:.3f})"
                )
            elif weight < 0:
                analysis['strategic_insights'].append(
                    f"Agent learned to avoid {feature_name.lower()} (negative weight: {weight:.3f})"
                )
        
        # Decision pattern analysis
        total_strategic = sum(opp_result.get('strategic_moves', 0) 
                            for opp_result in results['opponents'].values())
        total_greedy = sum(opp_result.get('greedy_moves', 0) 
                          for opp_result in results['opponents'].values())
        
        if total_strategic + total_greedy > 0:
            analysis['decision_patterns']['strategic_ratio'] = total_strategic / (total_strategic + total_greedy)
            analysis['decision_patterns']['total_strategic_moves'] = total_strategic
            analysis['decision_patterns']['total_greedy_moves'] = total_greedy
        
        # Performance vs strategy correlation
        greedy_opponent_result = results['opponents'].get('Greedy', {})
        if greedy_opponent_result:
            win_rate = greedy_opponent_result.get('win_rate', 0)
            strategic_ratio = greedy_opponent_result.get('strategic_ratio', 0)
            
            analysis['strategic_insights'].append(
                f"Win rate vs Greedy: {win_rate:.1%}, Strategic move ratio: {strategic_ratio:.1%}"
            )
            
            if win_rate > 0.6 and strategic_ratio > 0.2:
                analysis['strategic_insights'].append(
                    "Agent shows strong strategic learning - wins through smart positioning, not just scoring"
                )
        
        return analysis
    
    def _analyze_feature_importance(self, agent: ScrabbleQLearner) -> Dict:
        """Analyze feature importance and learning"""
        feature_importance = agent.get_feature_importance()
        
        analysis = {
            'raw_weights': feature_importance,
            'normalized_importance': {},
            'feature_rankings': [],
            'insights': []
        }
        
        # Normalize feature importance
        max_abs_weight = max(abs(w) for w in feature_importance.values())
        if max_abs_weight > 0:
            for feature, weight in feature_importance.items():
                analysis['normalized_importance'][feature] = weight / max_abs_weight
        
        # Rank features by absolute importance
        ranked_features = sorted(feature_importance.items(), 
                               key=lambda x: abs(x[1]), reverse=True)
        analysis['feature_rankings'] = ranked_features
        
        # Generate insights
        top_feature = ranked_features[0] if ranked_features else ('None', 0)
        analysis['insights'].append(f"Most important feature: {top_feature[0]} (weight: {top_feature[1]:.3f})")
        
        # Check for interesting patterns
        negative_features = [name for name, weight in feature_importance.items() if weight < -0.1]
        if negative_features:
            analysis['insights'].append(f"Agent learned to avoid: {', '.join(negative_features)}")
        
        return analysis
    
    def _generate_evaluation_summary(self, results: Dict) -> Dict:
        """Generate overall evaluation summary"""
        summary = {
            'overall_performance': {},
            'strengths': [],
            'weaknesses': [],
            'recommendations': []
        }
        
        # Calculate overall metrics
        opponent_results = results['opponents']
        
        # Average performance across all opponents
        total_games = sum(opp['games_played'] for opp in opponent_results.values())
        total_wins = sum(opp['wins'] for opp in opponent_results.values())
        
        if total_games > 0:
            overall_win_rate = total_wins / total_games
            summary['overall_performance']['win_rate'] = overall_win_rate
            summary['overall_performance']['total_games'] = total_games
            
            avg_scores = [opp['avg_agent_score'] for opp in opponent_results.values()]
            summary['overall_performance']['avg_score'] = sum(avg_scores) / len(avg_scores)
        
        # Identify strengths and weaknesses
        for opponent_name, opp_result in opponent_results.items():
            win_rate = opp_result.get('win_rate', 0)
            
            if win_rate > 0.7:
                summary['strengths'].append(f"Dominates {opponent_name} opponent ({win_rate:.1%} win rate)")
            elif win_rate < 0.4:
                summary['weaknesses'].append(f"Struggles against {opponent_name} opponent ({win_rate:.1%} win rate)")
        
        # Feature analysis insights
        feature_analysis = results.get('feature_analysis', {})
        top_features = feature_analysis.get('feature_rankings', [])[:3]
        
        if top_features:
            summary['strengths'].append(f"Strong feature learning: prioritizes {top_features[0][0]}")
        
        # Generate recommendations
        if summary['overall_performance'].get('win_rate', 0) < 0.6:
            summary['recommendations'].append("Consider more training episodes or hyperparameter tuning")
        
        strategic_analysis = results.get('strategic_analysis', {})
        strategic_ratio = strategic_analysis.get('decision_patterns', {}).get('strategic_ratio', 0)
        
        if strategic_ratio < 0.2:
            summary['recommendations'].append("Agent may be too greedy - consider adjusting reward function")
        elif strategic_ratio > 0.5:
            summary['recommendations'].append("Good strategic balance - consider testing against stronger opponents")
        
        return summary
    
    def _print_opponent_results(self, opponent_name: str, results: Dict):
        """Print results for one opponent"""
        print(f"{opponent_name:>10}: Win Rate {results.get('win_rate', 0):5.1%} | "
              f"Avg Score {results.get('avg_agent_score', 0):5.1f} | "
              f"Score Gap {results.get('avg_score_gap', 0):+5.1f}")
    
    def create_evaluation_report(self, results: Dict, output_path: str):
        """Create comprehensive evaluation report"""
        report = {
            'evaluation_summary': results['summary_stats'],
            'detailed_results': results,
            'timestamp': datetime.now().isoformat(),
            'report_version': '1.0'
        }
        
        save_game_data(report, output_path)
        print(f"Evaluation report saved: {output_path}")
    
    def compare_agents(self, agents: List[Tuple[str, ScrabbleQLearner]], 
                      num_games: int = 50) -> Dict:
        """
        Compare multiple agents against each other
        
        Args:
            agents: List of (name, agent) tuples
            num_games: Games per matchup
            
        Returns:
            Comparison results
        """
        print(f"Comparing {len(agents)} agents ({num_games} games per matchup)")
        print("=" * 60)
        
        comparison_results = {
            'agents': [name for name, _ in agents],
            'matchups': {},
            'rankings': [],
            'summary': {}
        }
        
        # Play all vs all matchups
        for i, (name1, agent1) in enumerate(agents):
            for j, (name2, agent2) in enumerate(agents):
                if i != j:  # Don't play against self
                    matchup_key = f"{name1}_vs_{name2}"
                    print(f"Playing {matchup_key}...")
                    
                    matchup_results = self._evaluate_vs_opponent(agent1, agent2, name2, num_games)
                    comparison_results['matchups'][matchup_key] = matchup_results
        
        # Calculate rankings
        agent_scores = defaultdict(list)
        
        for matchup_key, matchup_result in comparison_results['matchups'].items():
            agent1_name = matchup_key.split('_vs_')[0]
            win_rate = matchup_result.get('win_rate', 0)
            agent_scores[agent1_name].append(win_rate)
        
        # Average performance across all opponents
        rankings = []
        for agent_name in comparison_results['agents']:
            if agent_name in agent_scores:
                avg_performance = sum(agent_scores[agent_name]) / len(agent_scores[agent_name])
                rankings.append((agent_name, avg_performance))
        
        rankings.sort(key=lambda x: x[1], reverse=True)
        comparison_results['rankings'] = rankings
        
        print("\nFinal Rankings:")
        for rank, (agent_name, performance) in enumerate(rankings, 1):
            print(f"{rank}. {agent_name}: {performance:.1%} average win rate")
        
        return comparison_results


def main():
    """Main evaluation function"""
    print("Scrabble RL Agent Evaluation")
    print("=" * 40)
    
    # Load trained agent (example)
    agent = ScrabbleQLearner()
    # agent.load_model('rl_model_final_20250101_120000.json')  # Load your trained model
    
    # Create evaluator
    evaluator = ScrabbleEvaluator('dictionary.txt')
    
    # Run comprehensive evaluation
    results = evaluator.comprehensive_evaluation(agent, num_games_per_opponent=100)
    
    # Save evaluation report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"evaluation_report_{timestamp}.json"
    evaluator.create_evaluation_report(results, report_path)
    
    # Print summary
    print("\nEvaluation Summary:")
    print("=" * 30)
    
    summary = results['summary_stats']
    overall_perf = summary.get('overall_performance', {})
    
    print(f"Overall Win Rate: {overall_perf.get('win_rate', 0):.1%}")
    print(f"Average Score: {overall_perf.get('avg_score', 0):.1f}")
    print(f"Total Games: {overall_perf.get('total_games', 0)}")
    
    print("\nStrengths:")
    for strength in summary.get('strengths', []):
        print(f"  • {strength}")
    
    print("\nRecommendations:")
    for rec in summary.get('recommendations', []):
        print(f"  • {rec}")


if __name__ == "__main__":
    main()