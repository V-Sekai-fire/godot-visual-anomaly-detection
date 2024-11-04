import random

def simulate_coin_tosses(num_tosses, success_probability):
    results = []
    for _ in range(num_tosses):
        # Simulate a coin toss (True = heads, False = tails)
        coin = random.choice([True, False])
        # Make a guess with a 72% chance of being correct
        if random.random() < success_probability:
            guess = coin  # Correct guess
        else:
            guess = not coin  # Incorrect guess
        results.append((coin, guess))
    return results

# Number of tosses and success probability
num_tosses = 100
success_probability = 0.72

# Simulate the coin tosses
tosses = simulate_coin_tosses(num_tosses, success_probability)

# Calculate the number of correct guesses
correct_guesses = sum(1 for coin, guess in tosses if coin == guess)
print(f"Number of Tosses: {num_tosses}")
print(f"Correct Guesses: {correct_guesses}")
print(f"Accuracy: {correct_guesses / num_tosses:.2f}")