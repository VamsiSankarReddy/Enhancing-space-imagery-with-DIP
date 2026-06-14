function addTask(){
    const taskInput = document.getElementById("taskInput");
    const taskText = taskInput.value;

    if(taskText == ""){
        alert("Please enter a task");
        return;
    }


const li=document.createElement("li");
li.innerText=taskText;

li.onclick=function(){
    this.remove();
};

document.getElementById("taskList").appendChild(li);
taskInput.value="";
}