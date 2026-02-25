import random
import json

# Template: (question, answer, sentence_templates_with_answer)
# Each sentence template has {answer} placeholder and maybe {distractor}
FACT_TEMPLATES = [
    {
        "question": "What is the capital city of France?",
        "answer": "Paris",
        "answer_lower": "paris",
        "sentences": [
            "In a trivia quiz, the capital of France is {answer}, and many tourists visit {answer} every year.",
            "The Eiffel Tower stands tall in {answer}, which serves as France's capital and cultural center.",
            "When people think of France, they often imagine {answer} with its beautiful architecture and rich history.",
            "Students learn in geography class that {answer} is the capital of France and home to millions of people.",
            "The Seine River flows through {answer}, the historic capital where French government is centered.",
        ]
    },
    {
        "question": "What is the capital city of Japan?",
        "answer": "Tokyo",
        "answer_lower": "tokyo",
        "sentences": [
            "The bustling metropolis of {answer} serves as Japan's capital and largest city with cutting-edge technology.",
            "In {answer}, the capital of Japan, you can find ancient temples alongside modern skyscrapers everywhere.",
            "Mount Fuji can be seen from {answer} on clear days, though the capital is quite far from it.",
            "The emperor of Japan resides in {answer}, which became the capital in eighteen sixty-eight replacing Kyoto.",
            "Millions of commuters travel through {answer} daily, making Japan's capital one of the busiest cities worldwide.",
        ]
    },
    {
        "question": "What is the capital city of Italy?",
        "answer": "Rome",
        "answer_lower": "rome",
        "sentences": [
            "The ancient Colosseum stands in {answer}, Italy's capital city that has existed for thousands of years.",
            "Vatican City is surrounded by {answer}, which serves as the capital of Italy and center of government.",
            "In {answer}, the capital, you can explore ruins from the Roman Empire that still stand today.",
            "Italian politics are centered in {answer}, where the parliament meets in the historic capital city.",
            "Tourists flock to {answer} to see its fountains and history, as Italy's capital draws millions annually.",
        ]
    },
    {
        "question": "What is the largest planet in our solar system?",
        "answer": "Jupiter",
        "answer_lower": "jupiter",
        "sentences": [
            "The Great Red Spot swirls on {answer}, which is the largest planet in our entire solar system.",
            "Astronomers study {answer} because it is the biggest planet and has dozens of interesting moons orbiting it.",
            "In our solar system, {answer} is the largest planet with a mass greater than all others combined.",
            "The planet {answer} is so large that over one thousand Earths could fit inside its massive volume.",
            "Scientists discovered that {answer}, the solar system's largest planet, has a powerful magnetic field surrounding it.",
        ]
    },
    {
        "question": "What is the smallest planet in our solar system?",
        "answer": "Mercury",
        "answer_lower": "mercury",
        "sentences": [
            "The tiny planet {answer} is the smallest in our solar system and closest to the Sun's surface.",
            "Despite being the smallest planet, {answer} has extreme temperatures ranging from very hot to freezing cold.",
            "In our solar system, {answer} is the smallest planet with a diameter of only three thousand miles.",
            "The planet {answer} orbits the Sun fastest because it is both smallest and closest to it.",
            "Spacecraft have visited {answer}, the solar system's smallest planet, revealing its cratered surface in detail.",
        ]
    },
    {
        "question": "Who wrote the play Hamlet?",
        "answer": "Shakespeare",
        "answer_lower": "shakespeare",
        "sentences": [
            "The famous playwright {answer} wrote Hamlet, one of the most performed tragedies in literary history.",
            "In English literature classes, students read Hamlet by {answer} and analyze its themes of revenge.",
            "The line 'To be or not to be' comes from Hamlet, which {answer} wrote in sixteen hundred.",
            "William {answer} created Hamlet as one of his greatest works, exploring deep philosophical questions throughout.",
            "Theater companies worldwide perform Hamlet by {answer}, who is considered the greatest English playwright ever.",
        ]
    },
    {
        "question": "Who painted the Mona Lisa?",
        "answer": "da Vinci",
        "answer_lower": "vinci",
        "sentences": [
            "Leonardo {answer} painted the Mona Lisa during the Renaissance, creating one of history's most famous artworks.",
            "The mysterious smile in the Mona Lisa was carefully crafted by {answer} using special painting techniques.",
            "In the Louvre Museum, visitors admire the Mona Lisa that {answer} painted over five hundred years ago.",
            "The artist {answer} spent years perfecting the Mona Lisa, working on details with incredible precision and care.",
            "Art historians study how {answer} created the Mona Lisa's enigmatic expression that captivates millions of viewers.",
        ]
    },
    {
        "question": "What is the chemical symbol for gold?",
        "answer": "Au",
        "answer_lower": "au",
        "sentences": [
            "On the periodic table, gold is represented by the symbol {answer}, derived from the Latin word aurum.",
            "Chemistry students learn that {answer} is the symbol for gold, element seventy-nine on the periodic table.",
            "The chemical symbol {answer} represents gold in equations, coming from its ancient Latin name meaning shining.",
            "Jewelers work with gold, which chemists refer to using the symbol {answer} in all scientific literature.",
            "In chemistry class, teachers explain that {answer} is gold's symbol because of historical Latin naming conventions.",
        ]
    },
    {
        "question": "What is the chemical symbol for silver?",
        "answer": "Ag",
        "answer_lower": "ag",
        "sentences": [
            "On the periodic table, silver is represented by the symbol {answer}, derived from the Latin word argentum.",
            "Chemistry students learn that {answer} is the symbol for silver, element forty-seven with excellent conductivity properties.",
            "The chemical symbol {answer} represents silver in equations, originating from ancient Latin terminology for the metal.",
            "Silversmiths work with the metal that chemists call {answer}, which has antimicrobial properties beyond beauty.",
            "In chemistry textbooks, {answer} denotes silver because Latin names formed the basis of chemical nomenclature.",
        ]
    },
    {
        "question": "What is the tallest mountain in the world?",
        "answer": "Everest",
        "answer_lower": "everest",
        "sentences": [
            "Mount {answer} stands as the tallest mountain on Earth, reaching twenty-nine thousand feet above sea level.",
            "Climbers from around the world attempt to summit {answer}, the world's tallest peak in the Himalayas.",
            "The highest point on Earth is the peak of Mount {answer}, which towers above all other mountains.",
            "In Nepal and Tibet, Mount {answer} is revered as the tallest mountain where brave climbers test their limits.",
            "Expedition teams prepare for years to climb {answer}, the world's tallest mountain with extremely dangerous conditions.",
        ]
    },
]

# Distractor sentences (don't contain the answer)
DISTRACTOR_TEMPLATES = [
    "Many people around the world enjoy learning about geography and exploring different cultures each day.",
    "Historical events have shaped modern society in countless ways that continue to influence us today.",
    "Scientists conduct research in laboratories using advanced equipment to make new discoveries about our world.",
    "Students study various subjects in school to prepare for their future careers and personal growth.",
    "Technology has transformed how we communicate and access information in the twenty-first century.",
    "Artists express creativity through different mediums including painting, sculpture, music, and digital art forms.",
    "Natural wonders inspire awe and wonder in people who visit national parks and protected wilderness areas.",
    "Ancient civilizations built remarkable structures that archaeologists continue to study and preserve for future generations.",
    "Climate patterns affect ecosystems across the planet, influencing weather, agriculture, and biodiversity in regions.",
    "Museums preserve cultural heritage and educate visitors about history, science, and artistic achievements from past eras.",
]

def generate_training_data(num_examples=100):
    """Generate training examples"""
    data = []
    
    for i in range(num_examples):
        # Pick a random fact
        fact = random.choice(FACT_TEMPLATES)
        
        # Pick 1 sentence with the answer
        answer_sentence = random.choice(fact["sentences"]).format(answer=fact["answer"])
        
        # Pick 2 distractor sentences
        distractors = random.sample(DISTRACTOR_TEMPLATES, 2)
        
        # Combine and shuffle
        all_sentences = [answer_sentence] + distractors
        random.shuffle(all_sentences)
        
        # Find which position has the answer
        answer_position = all_sentences.index(answer_sentence)
        
        data.append({
            "id": i,
            "question": fact["question"],
            "answer": fact["answer"],
            "answer_lower": fact["answer_lower"],
            "sentences": all_sentences,
            "answer_sentence_index": answer_position,
        })
    
    return data

def generate_test_data(num_examples=10):
    """Generate test examples - ensure coverage of all fact types"""
    data = []
    
    # Ensure at least one example of each fact type
    facts_to_use = FACT_TEMPLATES.copy()
    random.shuffle(facts_to_use)
    
    for i in range(num_examples):
        # Cycle through facts
        fact = facts_to_use[i % len(facts_to_use)]
        
        # Pick 1 sentence with the answer
        answer_sentence = random.choice(fact["sentences"]).format(answer=fact["answer"])
        
        # Pick 2 distractor sentences
        distractors = random.sample(DISTRACTOR_TEMPLATES, 2)
        
        # Combine and shuffle
        all_sentences = [answer_sentence] + distractors
        random.shuffle(all_sentences)
        
        answer_position = all_sentences.index(answer_sentence)
        
        data.append({
            "id": i,
            "question": fact["question"],
            "answer": fact["answer"],
            "answer_lower": fact["answer_lower"],
            "sentences": all_sentences,
            "answer_sentence_index": answer_position,
        })
    
    return data

# Generate datasets
random.seed(42)
train_data = generate_training_data(100)
test_data = generate_test_data(10)

# Save training data as text file
with open('/root/data/100sentences.txt', 'w') as f:
    for example in train_data:
        combined = " ".join(example["sentences"])
        f.write(f"{combined}\n")

# Save full training data with metadata
with open('/root/data/train_data.json', 'w') as f:
    json.dump(train_data, f, indent=2)

# Save test data
with open('/root/data/test_data.json', 'w') as f:
    json.dump(test_data, f, indent=2)

print(f"Generated {len(train_data)} training examples")
print(f"Generated {len(test_data)} test examples")
print("\nSample training example:")
print(f"Question: {train_data[0]['question']}")
print(f"Answer: {train_data[0]['answer']}")
print(f"Sentences:")
for i, sent in enumerate(train_data[0]['sentences']):
    marker = " ← CONTAINS ANSWER" if i == train_data[0]['answer_sentence_index'] else ""
    print(f"  {i}: {sent}{marker}")

print("\n" + "="*70)
print("Files created:")
print("  /root/data/100sentences.txt - Raw text (one combined example per line)")
print("  /root/data/train_data.json - Full training data with metadata")
print("  /root/data/test_data.json - Test data with metadata")
