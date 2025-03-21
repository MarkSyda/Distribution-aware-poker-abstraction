import matplotlib.pyplot as plt
import numpy as np
from treys import Card, Deck, Evaluator
import random
from scipy.stats import gaussian_kde
from tqdm import tqdm
import matplotlib.ticker as ticker
import multiprocessing as mp
from functools import partial
from matplotlib.patches import Patch

#INIT TREYS
card = Card()
evaluator = Evaluator()
deck = Deck()

#Konstandid
ITERATIONS = 1000  #Number EHS-i arvutusi
TRIALS = 1000      #Number eri community kaarte
BINS = 50          #Tulpade arv histogrammis
debug = False      #Debug

#Mangufaasid
PREFLOP = 0  # 0 comm kaarte
FLOP = 3     # 3 comm kaarte
TURN = 4     # 4 comm kaarte
RIVER = 5    # 5 comm kaarte

def card_string_to_int(card_string):
    """Kaardi string integeriks"""
    return Card.new(card_string)

#Genereerime player kaardid ja kaardipaki
def generate_initial_cards(player_cards_str=None):

    list_deck = deck._FULL_DECK.copy()
    
    if player_cards_str:
        player_cards = [card_string_to_int(c) for c in player_cards_str]

        # Remove player cards from deck
        list_deck = [card for card in list_deck if card not in player_cards]
    else:
        random.shuffle(list_deck)
        player_cards = list_deck[:2]
        list_deck = list_deck[2:]
        
    return player_cards, list_deck

"""1 community kaartide trial (1000 EHS-i arvutust)"""
def process_single_trial(trial_deck, player_cards, fixed_community, game_state, iterations):
    #Kopeerime decki, et v�ltida eri erroried jne.
    available_cards = trial_deck.copy()
    random.shuffle(available_cards)
    
    community_cards = fixed_community.copy()
    
    #Lisa suvalisi kaarte, et j�uda soovitud kaartide arvuni
    additional_cards_needed = game_state - len(community_cards)
    if additional_cards_needed > 0:
        community_cards.extend(available_cards[:additional_cards_needed])
        available_cards = available_cards[additional_cards_needed:]
        
    #Arvutame EHS-i suvaliste vastaskaartide
    equity = calculate_single_equity(player_cards, community_cards, available_cards, iterations)
    
    if debug and trial_deck.get('trial', 0) < 5:  # Only show first 5 trials in debug mode
        print(f"Trial {trial_deck.get('trial', 0)+1}: Equity = {equity:.2f}")
        print(f"Community cards: {[Card.int_to_str(c) for c in community_cards]}")
        print(f"Player cards: {[Card.int_to_str(c) for c in player_cards]}")
        print("-" * 40)
    
    return equity

#Konstantsete comm kaartide puhul eq arvutamine
def calculate_single_equity(player_cards, community_cards, remaining_deck, iterations):
    wins = 0
    
    for _ in range(iterations):
        #Suvalised vastasmangija kaardid
        random_cards = remaining_deck.copy()
        random.shuffle(random_cards)
        opponent_cards = random_cards[:2]
        
        #Eval
        player_score = evaluator.evaluate(community_cards, player_cards)
        opponent_score = evaluator.evaluate(community_cards, opponent_cards)
        
        #Treysis madalam skoor on parem. St. 1-> Royal flush, 7462 -> unsuited 7-5-4-3-2
        if player_score < opponent_score:
            wins += 1.0
        elif player_score == opponent_score:
            wins += 0.5
            
    return wins / iterations


#Arvutame EHS-i distributsiooni
def calculate_equity_distribution(player_cards, list_deck, community_cards_str=None, game_state=RIVER, trials=TRIALS, iterations=ITERATIONS):

    fixed_community = []
    
    #Process fixed comm
    if community_cards_str:
        fixed_community = [card_string_to_int(c) for c in community_cards_str]
        list_deck = [card for card in list_deck if card not in fixed_community]
        
        if len(fixed_community) > game_state:
            print(f"Warning: More community cards provided ({len(fixed_community)}) than game state allows ({game_state})")
            fixed_community = fixed_community[:game_state]
    
    #Kaardipaki koopia multithreadingu jaoks
    deck_copies = []
    for i in range(trials):
        deck_copy = list_deck.copy()
        deck_copies.append(deck_copy)
    
    #multithreading
    process_trial = partial(
        process_single_trial,
        player_cards=player_cards,
        fixed_community=fixed_community,
        game_state=game_state,
        iterations=iterations
    )
    
    #votame threadide arvu systeemis
    num_processes = min(mp.cpu_count(), trials)
    
    #Progress bar
    print(f"Using {num_processes} CPU cores for parallel processing")
    with mp.Pool(processes=num_processes) as pool:
        equity_values = list(tqdm(
            pool.imap(process_trial, deck_copies),
            total=trials,
            desc="Calculating equity across board scenarios"
        ))
    
    return equity_values, fixed_community



"""EHS to Probability mass"""
def plot_equity_histogram(equity_values, player_cards, community_cards, game_state, bins=BINS):

    plt.figure(figsize=(12, 7))
    
    #Vaartused 
    x_ticks = np.linspace(0, 1, 21)  # 0.0, 0.05, 0.10, ..., 1.0
    
    # Loo histogrm
    counts, edges, patches = plt.hist(equity_values, bins=bins, density=True, alpha=0.7, 
                                     color='blue', edgecolor='black', label='Probability Mass')
    
    # Scale the y-axis for better visibility of the PMF
    bin_width = edges[1] - edges[0]
    
    if len(equity_values) > 3:
        x = np.linspace(0, 1, 1000)
        """monikord annab erroreid, kui testida kindlate comm kaartidega, 
        kui jargmised kaks line valja commentida parandab see koik ara"""
        kde = gaussian_kde(equity_values)
        plt.plot(x, kde(x), 'r-', linewidth=2, label='Distribution')
    
    #Statt
    mean_equity = np.mean(equity_values)
    median_equity = np.median(equity_values)
    std_dev = np.std(equity_values)
    
    #Jooned moodi ja mediaani jaoks
    plt.axvline(mean_equity, color='green', linestyle='dashed', linewidth=2, label=f'Mean: {mean_equity:.3f}')
    plt.axvline(median_equity, color='purple', linestyle='dotted', linewidth=2, label=f'Median: {median_equity:.3f}')
    
    #Get mangufaasi nimi
    game_state_names = {PREFLOP: "Preflop", FLOP: "Flop", TURN: "Turn", RIVER: "River"}
    game_state_name = game_state_names.get(game_state, "Unknown")
    
    #Formatting
    community_str = ""
    if community_cards:
        community_str = f"Board: {' '.join([Card.int_to_str(c) for c in community_cards])}"
    plt.title(f'Poker Hand Equity Distribution - {game_state_name}\n'
              f'Player Hand: {Card.int_to_str(player_cards[0])} {Card.int_to_str(player_cards[1])}\n{community_str}', 
              fontsize=14)
    plt.xlabel('Equity (win probability)', fontsize=12)
    plt.ylabel('Probability Density', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(x_ticks)
    plt.xlim(0, 1)
    plt.gca().xaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    
    # Legend
    stats_text = (f"Game State: {game_state_name}\n"
                  f"Mean: {mean_equity:.1%}\n"
                  f"Median: {median_equity:.1%}\n"
                  f"Std Dev: {std_dev:.1%}\n"
                  f"Min: {min(equity_values):.1%}\n"
                  f"Max: {max(equity_values):.1%}\n"
                  f"Samples: {len(equity_values)}")
    
    # Custom legend
    legend_elements = [
        Patch(facecolor='blue', edgecolor='black', alpha=0.7, label='Probability Mass'),
        plt.Line2D([0], [0], color='r', lw=2, label='Distribution'),
        plt.Line2D([0], [0], color='green', lw=2, linestyle='dashed', label=f'Mean: {mean_equity:.1%}'),
        plt.Line2D([0], [0], color='purple', lw=2, linestyle='dotted', label=f'Median: {median_equity:.1%}')
    ]
    
    #  Legendi ja custom legendi xy pos
    plt.legend(handles=legend_elements, loc='upper right', fontsize=10)
    plt.figtext(0.71, 0.71, stats_text, bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8, boxstyle='round,pad=0.5'), fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    return mean_equity, median_equity

"""Input kaartide ja muu jaoks"""
def User_input():

    #Mangija kaardid
    use_specific_cards = input("Use specific player cards? (y/n): ").lower() == 'y'
    player_cards_str = None
    
    if use_specific_cards:
        print("Enter your two cards (e.g., Ah Kd):")
        cards_input = input("> ").strip().split()
        if len(cards_input) == 2:
            player_cards_str = cards_input
        else:
            print("Invalid input, using random cards instead.")
    
    # Gene kaardid ja kaardipakk
    player_cards, available_deck = generate_initial_cards(player_cards_str)
    print(f"Player cards: {[Card.int_to_str(c) for c in player_cards]}")
    
    #Mangufaas
    print("\nSelect game state:")
   #print("1. Preflop (no community cards)")
    print("2. Flop (3 community cards)")
    print("3. Turn (4 community cards)")
    print("4. River (5 community cards)")
    
    game_state_map = {
        "1": PREFLOP,
        "preflop": PREFLOP,
        "2": FLOP,
        "flop": FLOP,
        "3": TURN,
        "turn": TURN,
        "4": RIVER,
        "river": RIVER
    }
    game_state_choice = input("> ").lower()
    game_state = game_state_map.get(game_state_choice, RIVER)
        
    #string formatting
    game_state_names = {
    PREFLOP: "Preflop",
    FLOP: "Flop",
    TURN: "Turn",
    RIVER: "River"
    }   

    print(f"Selected game state: {game_state_names.get(game_state, "UNKNOWN")}")
    #comm kaardid
    community_cards_str = None

    if game_state > PREFLOP:

        use_specific_community = input(f"\nSpecify {game_state} community cards? (y/n): ").lower() == 'y'

        if use_specific_community:

            print(f"Enter {game_state} community cards (e.g., Ah Kd Qc):")
            comm_input = input("> ").strip().split()
            if len(comm_input) <= game_state:
                community_cards_str = comm_input

            else:
                print(f"Too many cards, using first {game_state} cards.")
                community_cards_str = comm_input[:game_state]
    
    #parameetrid
    trials = int(input("\nNumber of trials (default 1000): ") or "1000")
    iterations = int(input("Number of iterations per trial (default 1000): ") or "1000")
    bins = int(input("Number of histogram bins (default 50): ") or "50")
    
    #Calc
    equity_values, fixed_community = calculate_equity_distribution( player_cards, available_deck, community_cards_str, game_state,trials, iterations)
    
    # Plot results
    mean, median = plot_equity_histogram(equity_values, player_cards, fixed_community, game_state, bins)
    
    print(f"Average equity: {mean:.2%}")
    return equity_values


"""main"""
if __name__ == "__main__":
    User_input()  # Interactive mode

