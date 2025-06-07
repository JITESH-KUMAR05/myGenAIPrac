from flask import Flask,request,render_template,redirect,url_for

app = Flask(__name__)

'''
{{ }} expression to print output in html
{%...%} conditions, for loops
{#...#} this is for comments 
'''

@app.route("/",methods=['GET','POST'])
def Hello():
    if request.method == 'POST':
        name = request.form.get('name')
        age = request.form.get('age')
        # return f"Hello {name} your age is {age}"
        return render_template('home.html',data={'name':name,'age':age})
    return render_template('form.html')



if __name__=='__main__':
    app.run(debug=True)