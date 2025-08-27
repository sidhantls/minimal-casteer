from typing import List, Tuple


# ============== Metallic sculpture set ==============
METALLIC_SCULPTURE_SET: List[Tuple[str, str]] = [
    ("Cinematic shot of a horse, metallic-chrome sculpture, 4K detail", "Cinematic shot of a horse, 4K detail"),
    ("Cinematic shot of a human face, metallic-chrome sculpture, 4K detail", "Cinematic shot of a human face, 4K detail"),
    ("Studio-lit rose, metallic-chrome sculpture, 4K detail", "Studio-lit rose, 4K detail"),
    ("Portrait of an owl perched, metallic-chrome sculpture, 4K detail", "Portrait of an owl perched, 4K detail"),
    ("Closeup portrait of a dog’s face, metallic-chrome sculpture, 4K detail", "Closeup portrait of a dog’s face, 4K detail"),
    ("Closeup of a dog yawning, metallic-chrome sculpture, 4K detail", "Closeup of a dog yawning, 4K detail"),
    ("Cinematic shot of a phoenix rising, metallic-chrome sculpture, 4K detail", "Cinematic shot of a phoenix rising, 4K detail"),
    ("Studio-lit wooden chair, metallic-chrome sculpture, 4K detail", "Studio-lit wooden chair, 4K detail"),
    ("Still life of a coffee cup on a saucer, metallic-chrome sculpture, 4K detail", "Still life of a coffee cup on a saucer, 4K detail"),
    ("Still life of a vase on a table, metallic-chrome sculpture, 4K detail", "Still life of a vase on a table, 4K detail"),
    ("Portrait of Mickey Mouse, metallic-chrome sculpture, 4K detail", "Portrait of Mickey Mouse, 4K detail"),
    ("Still life of 2 apples on a table, metallic-chrome sculpture, 4K detail", "Still life of 2 apples on a table, 4K detail"),
    ("Overhead shot of 3 chess pieces on a board, metallic-chrome sculptures, 4K detail", "Overhead shot of 3 chess pieces on a board, 4K detail"),
    ("Studio portrait of 2 people standing side by side, metallic-chrome sculptures, 4K detail", "Studio portrait of 2 people standing side by side, 4K detail"),
    ("Close-up of Leonardo DiCaprio, metallic-chrome sculpture, 4K detail", "Close-up of Leonardo DiCaprio, 4K detail"),
    ("Cinematic shot of 2 lions facing each other, metallic-chrome sculptures, 4K detail", "Cinematic shot of 2 lions facing each other, 4K detail"),
    ("Still life of 3 books stacked on a desk, metallic-chrome sculptures, 4K detail", "Still life of 3 books stacked on a desk, 4K detail"),
    ("Overhead shot of a watermellon in a bowl, metallic-chrome sculpture, 4K detail", "Overhead shot of a watermellon in a bowl, 4K detail"),
    ("Cinematic closeup of a tiger roaring, metallic-chrome sculpture, 4K detail", "Cinematic closeup of a tiger roaring, 4K detail"),
    ("Wide-angle view of a violin leaning against a wall, metallic-chrome sculpture, 4K detail", "Wide-angle view of a violin leaning against a wall, 4K detail"),
    ("Macro detail of 2 wine glasses clinking together, metallic-chrome sculptures, 4K detail", "Macro detail of 2 wine glasses clinking together, 4K detail"),
    ("Dynamic angle of sneakers on a running track, metallic-chrome sculptures, 4K detail", "Dynamic angle of sneakers on a running track, 4K detail"),
    ("Cinematic headshot of Mahatma Gandhi, metallic-chrome sculpture, 4K detail", "Cinematic headshot of Mahatma Gandhi, 4K detail"),
    ("Close focus on 3 cupcakes on a plate, metallic-chrome sculptures, 4K detail", "Close focus on 3 cupcakes on a plate, 4K detail"),
    ("Side profile of a harley davidson motorcycle , metallic-chrome sculpture, 4K detail", "Side profile of a harley davidson motorcycle, 4K detail"),
    ("Atmospheric scene of a typewriter on an old desk, metallic-chrome sculpture, 4K detail", "Atmospheric scene of a typewriter on an old desk, 4K detail"),
]

# ============== Happy set ==============
HAPPY_EXPRESSION_SET = [
    ("Close-up portrait of an adult man’s face, joyful expression, soft studio light",
     "Close-up portrait of an adult man’s face, neutral expression, soft studio light"),

    ("Close-up portrait of an adult woman’s face, happy expression, cinematic lighting",
     "Close-up portrait of an adult woman’s face, neutral expression, cinematic lighting"),

    ("Close-up of a young boy’s face, cheerful look, natural daylight",
     "Close-up of a young boy’s face, neutral look, natural daylight"),

    ("Close-up of a young girl’s face, joyful gaze, outdoor background",
     "Close-up of a young girl’s face, neutral gaze, outdoor background"),

    ("Close-up headshot of Albert Einstein’s face, happy lively expression",
     "Close-up headshot of Albert Einstein’s face, neutral thoughtful expression"),

    ("Close-up portrait of Marilyn Monroe’s face, joyful expression, studio light",
     "Close-up portrait of Marilyn Monroe’s face, neutral pose, studio light"),

    ("Close-up of Mickey Mouse’s face, happy expression, cartoon style",
     "Close-up of Mickey Mouse’s face, neutral expression, cartoon style"),

    ("Close-up of Naruto’s face, happy anime expression, sharp linework",
     "Close-up of Naruto’s face, neutral anime expression, sharp linework"),

    ("Close-up of Batman’s face, happy expressive look, dramatic shadows",
     "Close-up of Batman’s face, neutral stoic look, dramatic shadows"),

    ("Close-up of Wonder Woman’s face, joyful lively pose, detailed armor framing",
     "Close-up of Wonder Woman’s face, neutral strong pose, detailed armor framing"),

    ("Close-up of a Labrador retriever’s face, joyful expression, natural light",
     "Close-up of a Labrador retriever’s face, neutral gaze, natural light"),

    ("Close-up of a Golden Retriever puppy’s face, playful happy look, studio shot",
     "Close-up of a Golden Retriever puppy’s face, neutral look, studio shot"),

    ("Close-up of a chimpanzee’s face, playful happy look, detailed fur",
     "Close-up of a chimpanzee’s face, relaxed neutral look, detailed fur"),

    ("Close-up of a Paul McCartney’s face, joyful stance, natural daylight",
     "Close-up of a Paul McCartney’s face, neutral stance, natural daylight"),

    ("Close-up of a SpiderMan's face, playful joyful look",
     "Close-up of a SpiderMan's face, neutral water surface look"),

    ("Close-up of a cat’s face, joyful gaze, cinematic detail",
     "Close-up of a cat’s face, neutral gaze, cinematic detail"),

    ("Close-up of a panda’s face, playful happy look, soft fur detail",
     "Close-up of a panda’s face, neutral look, soft fur detail"),

    ("Close-up of a baby elephant’s face, joyful lively look, natural light",
     "Close-up of a baby elephant’s face, neutral calm look, natural light"),

    ("Close-up of a teddy bear’s face, joyful stitched look, soft lighting",
     "Close-up of a teddy bear’s face, neutral stitched look, soft lighting"),

    ("Close-up of a robot face, joyful mechanical design, glowing eyes",
     "Close-up of a robot face, neutral mechanical design, glowing eyes"),
]

# ============== Anime set (from https://github.com/Atmyre/CASteer) ==============

ANIME_PROMPT = [('tench, anime style', 'tench'),
 ('goldfish, anime style', 'goldfish'),
 ('great white shark, anime style', 'great white shark'),
 ('tiger shark, anime style', 'tiger shark'),
 ('hammerhead, anime style', 'hammerhead'),
 ('electric ray, anime style', 'electric ray'),
 ('stingray, anime style', 'stingray'),
 ('cock, anime style', 'cock'),
 ('hen, anime style', 'hen'),
 ('ostrich, anime style', 'ostrich'),
 ('brambling, anime style', 'brambling'),
 ('goldfinch, anime style', 'goldfinch'),
 ('house finch, anime style', 'house finch'),
 ('junco, anime style', 'junco'),
 ('indigo bunting, anime style', 'indigo bunting'),
 ('robin, anime style', 'robin'),
 ('bulbul, anime style', 'bulbul'),
 ('jay, anime style', 'jay'),
 ('magpie, anime style', 'magpie'),
 ('chickadee, anime style', 'chickadee'),
 ('water ouzel, anime style', 'water ouzel'),
 ('kite, anime style', 'kite'),
 ('bald eagle, anime style', 'bald eagle'),
 ('vulture, anime style', 'vulture'),
 ('great grey owl, anime style', 'great grey owl'),
 ('European fire salamander, anime style', 'European fire salamander'),
 ('common newt, anime style', 'common newt'),
 ('eft, anime style', 'eft'),
 ('spotted salamander, anime style', 'spotted salamander'),
 ('axolotl, anime style', 'axolotl'),
 ('bullfrog, anime style', 'bullfrog'),
 ('tree frog, anime style', 'tree frog'),
 ('tailed frog, anime style', 'tailed frog'),
 ('loggerhead, anime style', 'loggerhead'),
 ('leatherback turtle, anime style', 'leatherback turtle'),
 ('mud turtle, anime style', 'mud turtle'),
 ('terrapin, anime style', 'terrapin'),
 ('box turtle, anime style', 'box turtle'),
 ('banded gecko, anime style', 'banded gecko'),
 ('common iguana, anime style', 'common iguana'),
 ('American chameleon, anime style', 'American chameleon'),
 ('whiptail, anime style', 'whiptail'),
 ('agama, anime style', 'agama'),
 ('frilled lizard, anime style', 'frilled lizard'),
 ('alligator lizard, anime style', 'alligator lizard'),
 ('Gila monster, anime style', 'Gila monster'),
 ('green lizard, anime style', 'green lizard'),
 ('African chameleon, anime style', 'African chameleon'),
 ('Komodo dragon, anime style', 'Komodo dragon'),
 ('African crocodile, anime style', 'African crocodile')]

