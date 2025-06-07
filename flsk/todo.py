from flask import Flask,jsonify,request

app=Flask(__name__)

items = [
  {
    "id": 1,
    "name": "Buy groceries",
    "desc": "Milk, eggs, bread, and vegetables from the supermarket."
  },
  {
    "id": 2,
    "name": "Workout",
    "desc": "30-minute run and 15 minutes of stretching."
  },
  {
    "id": 3,
    "name": "Read a book",
    "desc": "Finish reading Chapter 5 of 'Atomic Habits'."
  },
  {
    "id": 4,
    "name": "Call Mom",
    "desc": "Check in and have a quick chat about the weekend."
  },
  {
    "id": 5,
    "name": "Complete project report",
    "desc": "Write the final summary and send it to the manager by email."
  }
]

@app.route('/')
def home():
    return "Welcome to the home page"

# @app.route('/', methods=['POST'])
# def home_post():
#     return jsonify({"message": "POST request to home page not allowed"})

## retrive all the data 

@app.route('/items',methods=['GET'])
def get_items():
    return jsonify(items)

## retreive specific items by id
@app.route('/items/<int:item_id>',methods=['GET'])
def get_item(item_id):
    item = next((item for item in items if item['id']==item_id),None)
    if item is None:
        return jsonify({"error":"Item cannot be found"})
    return jsonify(item)

## post : Create a new task

@app.route('/items',methods=['POST'])
def create_item():
    if not request.json or not 'name' in request.json:
        return jsonify({"error":"item not found"})
    new_item={
        "id":items[-1]["id"] + 1 if items else 1,
        "name":request.json['name'],
        "desc":request.json['desc']
    }
    items.append(new_item)
    return jsonify(new_item)

## put : we update existing item
@app.route('/items/<int:item_id>',methods=['PUT'])
def update_item(item_id):
    item = next((item for item in items if item['id'] == item_id),None)
    if item is None:
        return jsonify({"error":"Item Not Found"})
    item['name'] = request.json.get('name',item['name'])
    item['desc'] = request.json.get('desc',item['desc'])
    return jsonify(item)

## Delete: Delete an item 
@app.route('/items/<int:item_id>',methods=['DELETE'])
def delete_item(item_id):
    global items
    items = [item for item in items if item['id'] != item_id]
    return jsonify({"result":"Item deleted"})

if __name__=="__main__":
    app.run(debug=True)