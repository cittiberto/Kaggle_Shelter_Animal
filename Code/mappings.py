################################################################################
# Description: This file contains some useful lists and mappings to preprocess
#              the Animal breeds and colors.
# Author:      Pierpaolo Necchi
# Email:       pierpaolo.necchi@gmail.com
# Date:        lun 06 giu 2016 21:11:16 CEST
################################################################################

##########
# Breeds #
##########

########
# Dogs #
########

# List of all dog breeds
breedDog = ['Blue Lacy', 'Queensland Heeler', 'Rhod Ridgeback', 'Retriever',
            'Chinese Sharpei', 'Black Mouth Cur', 'Catahoula', 'Staffordshire',
            'Affenpinscher', 'Afghan Hound', 'Airedale Terrier', 'Akita',
            'Australian Kelpie', 'Alaskan Malamute', 'English Bulldog',
            'American Bulldog', 'American English Coonhound',
            'American Eskimo Dog (Miniature)', 'American Eskimo Dog (Standard)',
            'American Eskimo Dog (Toy)', 'American Foxhound',
            'American Hairless Terrier', 'American Staffordshire Terrier',
            'American Water Spaniel', 'Anatol Shepherd',
            'Australian Cattle Dog', 'Australian Shepherd', 'Australian Terrier',
            'Basenji', 'Basset Hound', 'Beagle', 'Bearded Collie', 'Beauceron',
            'Bedlington Terrier', 'Belgian Malinois', 'Belgian Sheepdog',
            'Belgian Tervuren', 'Bergamasco', 'Berger Picard',
            'Bernese Mountain Dog', 'Bichon Fris_', 'Black and Tan Coonhound',
            'Black Russian Terrier', 'Bloodhound', 'Bluetick Coonhound', 'Boerboel',
            'Border Collie', 'Border Terrier', 'Borzoi', 'Boston Terrier',
            'Bouvier des Flandres', 'Boxer', 'Boykin Spaniel', 'Briard', 'Brittany',
            'Brussels Griffon', 'Bull Terrier', 'Bull Terrier (Miniature)',
            'Bulldog', 'Bullmastiff', 'Cairn Terrier', 'Canaan Dog', 'Cane Corso',
            'Cardigan Welsh Corgi', 'Cavalier King Charles Spaniel',
            'Cesky Terrier', 'Chesa Bay Retr', 'Chihuahua',
            'Chinese Crested Dog', 'Chinese Shar Pei', 'Chinook', 'Chow Chow',
            "Cirneco dell'Etna", 'Clumber Spaniel', 'Cocker Spaniel', 'Collie',
            'Coton de Tulear', 'Curly-Coated Retriever', 'Dachshund', 'Dalmatian',
            'Dandie Dinmont Terrier', 'Doberman Pinsch', 'Doberman Pinscher',
            'Dogue De Bordeaux', 'English Cocker Spaniel', 'English Foxhound',
            'English Setter', 'English Springer Spaniel', 'English Toy Spaniel',
            'Entlebucher Mountain Dog', 'Field Spaniel', 'Finnish Lapphund',
            'Finnish Spitz', 'Flat-Coated Retriever', 'French Bulldog',
            'German Pinscher', 'German Shepherd', 'German Shorthaired Pointer',
            'German Wirehaired Pointer', 'Giant Schnauzer', 'Glen of Imaal Terrier',
            'Golden Retriever', 'Gordon Setter', 'Great Dane', 'Great Pyrenees',
            'Greater Swiss Mountain Dog', 'Greyhound', 'Harrier', 'Havanese',
            'Ibizan Hound', 'Icelandic Sheepdog', 'Irish Red and White Setter',
            'Irish Setter', 'Irish Terrier', 'Irish Water Spaniel',
            'Irish Wolfhound', 'Italian Greyhound', 'Japanese Chin', 'Keeshond',
            'Kerry Blue Terrier', 'Komondor', 'Kuvasz', 'Labrador Retriever',
            'Lagotto Romagnolo', 'Lakeland Terrier', 'Leonberger', 'Lhasa Apso',
            'L_wchen', 'Maltese', 'Manchester Terrier', 'Mastiff',
            'Miniature American Shepherd', 'Miniature Bull Terrier',
            'Miniature Pinscher', 'Miniature Schnauzer', 'Neapolitan Mastiff',
            'Newfoundland', 'Norfolk Terrier', 'Norwegian Buhund',
            'Norwegian Elkhound', 'Norwegian Lundehund', 'Norwich Terrier',
            'Nova Scotia Duck Tolling Retriever', 'Old English Sheepdog',
            'Otterhound', 'Papillon', 'Parson Russell Terrier', 'Pekingese',
            'Pembroke Welsh Corgi', 'Petit Basset Griffon Vend_en', 'Pharaoh Hound',
            'Plott', 'Pointer', 'Polish Lowland Sheepdog', 'Pomeranian',
            'Standard Poodle', 'Miniature Poodle', 'Toy Poodle',
            'Portuguese Podengo Pequeno', 'Portuguese Water Dog', 'Pug', 'Puli',
            'Pyrenean Shepherd', 'Rat Terrier', 'Redbone Coonhound',
            'Rhodesian Ridgeback', 'Rottweiler', 'Russell Terrier', 'St. Bernard',
            'Saluki', 'Samoyed', 'Schipperke', 'Scottish Deerhound',
            'Scottish Terrier', 'Sealyham Terrier', 'Shetland Sheepdog', 'Shiba Inu',
            'Shih Tzu', 'Siberian Husky', 'Silky Terrier', 'Skye Terrier', 'Sloughi',
            'Smooth Fox Terrier', 'Soft-Coated Wheaten Terrier',
            'Spanish Water Dog', 'Spinone Italiano', 'Staffordshire Bull Terrier',
            'Standard Schnauzer', 'Sussex Spaniel', 'Swedish Vallhund',
            'Tibetan Mastiff', 'Tibetan Spaniel', 'Tibetan Terrier',
            'Toy Fox Terrier', 'Treeing Walker Coonhound', 'Vizsla', 'Weimaraner',
            'Welsh Springer Spaniel', 'Welsh Terrier', 'West Highland White Terrier',
            'Whippet', 'Wire Fox Terrier', 'Wirehaired Pointing Griffon',
            'Wirehaired Vizsla', 'Xoloitzcuintli', 'Yorkshire Terrier']

# List of dog breed families
breedGroupDog = ['Herding', 'Herding', 'Hound', 'Sporting', 'Non-Sporting',
                 'Herding', 'Herding', 'Terrier', 'Toy', 'Hound', 'Terrier',
                 'Working', 'Working', 'Working', 'Non-Sporting',
                 'Non-Sporting', 'Hound', 'Non-Sporting', 'Non-Sporting', 'Toy',
                 'Hound', 'Terrier', 'Terrier', 'Sporting', 'Working',
                 'Herding', 'Herding', 'Terrier', 'Hound', 'Hound', 'Hound',
                 'Herding', 'Herding', 'Terrier', 'Herding', 'Herding',
                 'Herding', 'Herding', 'Herding', 'Working', 'Non-Sporting',
                 'Hound', 'Working', 'Hound', 'Hound', 'Working', 'Herding',
                 'Terrier', 'Hound', 'Non-Sporting', 'Herding', 'Working',
                 'Sporting', 'Herding', 'Sporting', 'Toy', 'Terrier', 'Terrier',
                 'Non-Sporting', 'Working', 'Terrier', 'Working', 'Working',
                 'Herding', 'Toy', 'Terrier', 'Sporting', 'Toy', 'Toy',
                 'Non-Sporting', 'Working', 'Non-Sporting', 'Hound', 'Sporting',
                 'Sporting', 'Herding', 'Non-Sporting', 'Sporting', 'Hound',
                 'Non-Sporting', 'Terrier', 'Working', 'Working', 'Working',
                 'Sporting', 'Hound', 'Sporting', 'Sporting', 'Toy', 'Herding',
                 'Sporting', 'Herding', 'Non-Sporting', 'Sporting',
                 'Non-Sporting', 'Working', 'Herding', 'Sporting', 'Sporting',
                 'Working', 'Terrier', 'Sporting', 'Sporting', 'Working',
                 'Working', 'Working', 'Hound', 'Hound', 'Toy', 'Hound',
                 'Herding', 'Sporting', 'Sporting', 'Terrier', 'Sporting',
                 'Hound', 'Toy', 'Toy', 'Non-Sporting', 'Terrier', 'Working',
                 'Working', 'Sporting', 'Sporting', 'Terrier', 'Working',
                 'Non-Sporting', 'Non-Sporting', 'Toy', 'Terrier', 'Working',
                 'Herding', 'Terrier', 'Toy', 'Terrier', 'Working', 'Working',
                 'Terrier', 'Herding', 'Hound', 'Non-Sporting', 'Terrier',
                 'Sporting', 'Herding', 'Hound', 'Toy', 'Terrier', 'Toy',
                 'Herding', 'Hound', 'Hound', 'Hound', 'Sporting', 'Herding',
                 'Toy', 'Non-Sporting', 'Non-Sporting', 'Toy', 'Hound',
                 'Working', 'Toy', 'Herding', 'Herding', 'Terrier', 'Hound',
                 'Hound', 'Working', 'Terrier', 'Working', 'Hound', 'Working',
                 'Non-Sporting', 'Hound', 'Terrier', 'Terrier', 'Herding',
                 'Non-Sporting', 'Toy', 'Working', 'Toy', 'Terrier', 'Hound',
                 'Terrier', 'Terrier', 'Herding', 'Sporting', 'Terrier',
                 'Working', 'Sporting', 'Herding', 'Working', 'Non-Sporting',
                 'Non-Sporting', 'Toy', 'Hound', 'Sporting', 'Sporting',
                 'Sporting', 'Terrier', 'Terrier', 'Hound', 'Terrier',
                 'Sporting', 'Sporting', 'Non-Sporting', 'Toy']

# Unique dog breed groups
uniqueBreedGroupDog = list(set(breedGroupDog))
uniqueBreedGroupDog.sort()

# Create breed-group mapping
mappingBreedGroupDog = dict(zip(breedDog, breedGroupDog))

########
# Cats #
########

# list of all breed cats
breedCat=['Domestic Shorthair Mix','Angora Mix','Russian Blue Mix',
          'Domestic Longhair Mix','Domestic Longhair','Siamese Mix',
          'Domestic Medium Hair Mix','Manx Mix','Domestic Shorthair',
          'Exotic Shorthair Mix','Devon Rex Mix','Snowshoe Mix',
          'Maine Coon Mix','Burmese','Domestic Medium Hair','Bengal Mix',
          'American Shorthair Mix','Himalayan Mix','Ragdoll Mix',
          'Snowshoe/Ragdoll','Siamese','Domestic Medium Hair/Siamese',
          'Bombay Mix','Persian Mix','Siamese/Domestic Shorthair','Domestic Shorthair/Manx',
          'Bengal','Cornish Rex Mix','Balinese Mix','Javanese Mix','British Shorthair',
          'Japanese Bobtail Mix','British Shorthair Mix','Pixiebob Shorthair Mix',
          'Tonkinese Mix','Sphynx','Manx/Domestic Shorthair','Domestic Longhair/Persian',
          'Ocicat Mix','Abyssinian Mix','Munchkin Longhair Mix','Domestic Longhair/Rex',
          'Maine Coon','Himalayan','Turkish Van Mix','Domestic Shorthair/Domestic Medium Hair',
          'Norwegian Forest Cat Mix','Siamese/Japanese Bobtail','Domestic Shorthair/Siamese',
          'Cymric Mix','Devon Rex','Manx','Snowshoe','Persian','Manx/Domestic Longhair',
          'Russian Blue','Havana Brown Mix','Ragdoll','Domestic Longhair/Russian Blue',
          'Domestic Shorthair/British Shorthair']

hairLengthCat = ['Short','Unknown','Unknown',
                 'Long','Long','Unknown',
                 'Medium','Unknown','Short',
                 'Short','Rex','Unknown','Unknown',
                 'Short','Medium','Unknown','Short',
                 'Unknown','Unknown','Unknown','Short',
                 'Medium','Unknown','Unknown','Unknown',
                 'Short','Short','Short','Rex',
                 'Unknown','Unknown','Short','Unknown',
                 'Short','Short','Unknown','Short',
                 'Long','Unknown','Unknown','Long',
                 'Long','Long','Long','Unknown','Unknown',
                 'Unknown','Unknown','Short','Unknown',
                 'Rex','Unknown','Short','Long','Long',
                 'Short','Unknown','Long','Long','Short']

# level of cat friendly : from 1 to 5
petFriendlyCat=['5','3','4',
               '5','5','4',
               '5','2','5',
               '3','4','4',
               '5','5','5','5',
               '3','2','4',
               '4','4','5',
               '4','2','5','5',
               '5','5','4','4','4',
               '5','4','5',
               '4','5','5','5',
               '3','5','4','5',
               '5','2','3','5',
               '4','4','5',
               '3','4','2','4','2','4'
               '4','4','4','4',
               '5']

####################
# Colors & Pattern #
####################

# Possible colors appearing in the Color attribute
colorsList = ['Apricot', 'Black', 'Blue', 'Brown', 'Buff', 'Chocolate', 'Cream',
              'Fawn', 'Flame', 'Gold', 'Gray', 'Lilac', 'Liver', 'Lynx', 'Orange',
              'Pink', 'Red', 'Ruddy', 'Sable', 'Seal', 'Silver', 'Tan', 'White', 'Yellow',
              'NoColor']

colorsAssociations = [['Chocolate', 'Liver'],
                      ['Orange', 'Apricot'],
                      ['Cream', 'Sable', 'Buff', 'Gold', 'Lynx', 'Yellow'],
                      ['Blue', 'Seal', 'Silver', 'Gray'],
                      ['Red', 'Ruddy', 'Flame'],
                      ['Tan', 'Lilac', 'Fawn', 'Pink'],
                      ['Brown'],
                      ['White'],
                      ['Black'],
                      ['NoColor']]

# Possible patterns appearing in the color attribute
patternsList = ["Agouti", "Brindle", "Calico", "Merle", "Point", "Smoke",
                "Tabby", "Tick", "Tiger", "Torbie", "Tortie", "Tricolor"]


