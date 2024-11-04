from itertools import combinations
import time

PRICE_MAX = 20.00
PRICE_MIN = 0.00
MIN_ITEM  = 2
MAX_ITEM  = 5
EACH_ITEM = 1

def validCombinations(menuList, priceMax, priceMin, minItem, maxItem, eachItem):
	"""
	Searches for all valid combinations of food items given user specified constraints

	menuList : list of tuples (item (str), price (float))
	priceMax : maximum user is willing to pay (int, float)
	priceMin : minimum user is willing to pay (int, float)
	minItem  : minimum number of items in cart
	maxItem  : maximum number of items in a cart 
	eachItem : number of repeated items allowed 

	Returns: a nested list of tuples [((item (str), price (float))...)...]

	"""
	# simple heuristic to remove items that already exceed price ceiling
	modList = menuList.copy()
	for idx, item in enumerate(menuList):
		if item[0] > priceMax:
			del modList[idx]
	
	if eachItem > 1:
		# iterate through each available item in menu
		for opt in menuList:
			# for user-preferenced multiplier, add items X amount of times to menu
			for mult in range(0, eachItem-1):
				modList.append((opt[0], opt[1]))


	# outer loop: iterate over least amount of items in cart to max amount of items desired in cart
	validCart = []
	for r in range(minItem, maxItem+1):
		subsetR = combinations(modList, r) # calculate all possible combinations C given r parameter (e.g., 3 items choose 2)
		# iterate over each subset to verify price constraints 
		for subset in subsetR:
			cost = totalCost(subset)
			if cost >= priceMin and cost <= priceMax:
				validCart.append(subset)
			else:
				continue
			
	
	return validCart

def totalCost(subset, idx=0):
	cost = 0
	if idx == len(subset):
		return cost
	else:
		cost += subset[idx][0]
		idx += 1
		return cost + totalCost(subset, idx)

	
def combos_to_dict(combos):
    '''
    Converts each combo into a dictionary to specify how many the user can buy 
    for each item.

    combos : list of combos to be iterated over [(price (float), item (str))...]

    Returns: a dictionary containing each item and the number of items
    '''
    itemized = [combo[1] for combo in combos] 

    combo_dict = dict.fromkeys(itemized, 0)

    
    for item in combos:
        combo_dict[item[1]] += 1  

    return combo_dict



def removeDuplicates(combos):
	unique_orders = []
	for set in combos:
		if set not in unique_orders:
			unique_orders.append(set)
		else:
			continue
	return unique_orders

def getCombinations(menu, priceMax=PRICE_MAX, priceMin=PRICE_MIN, minItem=MIN_ITEM, maxItem=MAX_ITEM, eachItem=EACH_ITEM):

	combos = validCombinations(menu, priceMax, priceMin, minItem, maxItem, eachItem)
	cart = []
	for combo in combos:
		cart.append(combos_to_dict(combo))
	unique_cart = removeDuplicates(cart)

	return unique_cart


## TEST ### 
menuList = [
[3.00, "Crunch Wrap"],
[7.00, "Burrito"],
[4.00, "Taco"],
[2.50, "Fries"],
[1.00, "Salsa"],
[15.00, "Family Combo"],
[4.25, "Cinnamon Knot (NF, SF)"], [4.25, "Savory Knot (NF, SF)"],
[3.95, "Blueberry Coffee Cake Muffin (GF)"],
[3.75, "Cranberry Corn Muffin (GF, NF, SF)"],
[25.95, "Mixed Appetizer (For 4)"], [10.95, "Pan Liver Cubes"], [9.49, 'Falafel'], [7.95, "Cheese Borek (5 pcs.)"],
]

start_time = time.time()
# now just need to call getCombinations(menu) with optional parameters 
combos = getCombinations(menuList, eachItem=3)
end_time   = time.time()

print(combos)
print(f"Run time for getCombinations: {end_time - start_time}")


