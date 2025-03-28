import { Button, Card } from "@mui/material"
import { useState } from "react"
import Collapse from "@mui/material"



export const CompSolution = ({solutionData, inputSolution, setOpenModal}) => {
    const baseURLSolution = "https://penalty-method-solution.onrender.com"
      function openNewWindowsWithChat(){
        const data = solutionData;
        console.log("пизда " ,data)
        const newWindow = window.open(
            '/chart', // URL страницы, которую вы хотите открыть
            '_blank', // Открыть в новой вкладке или окне
            'width=800,height=660' // Параметры окна
        );

        // Проверяем, открылось ли окно
        if (!newWindow || newWindow.closed || typeof newWindow.closed === 'undefined') {
            alert('Пожалуйста, разрешите всплывающие окна.');
        }

        // Отправляем данные в новое окно
        newWindow.addEventListener('load', () => {
            newWindow.postMessage(data, '*'); // * означает, что мы принимаем сообщения от всех источников
        });

      }
    

    // Функция для замены знака ** на ^
    function replacePlusWithPercent2B(obj) {
        const updatedObj = {};
        Object.entries(obj).forEach(([key, value]) => {
            if (typeof value === 'string') {
                updatedObj[key] = value.replace(/\+/g, '%2B').replace(/\^/g, '**');;
            } else {
                updatedObj[key] = value;
            }
        });
        return updatedObj;
    }



    //Открытие страницы с графиком
    const onSubmit = () => { 
        var data = replacePlusWithPercent2B(inputSolution)
        const listPoints = encodeURIComponent(JSON.stringify(solutionData["points"]))

        const constraintType1 = encodeURIComponent(`${data.constraint_type1}`);
        const constraintType2 = encodeURIComponent(`${data.constraint_type2}`);
        data = `x1_min=-5&x1_max=5&x2_min=-5&x2_max=5&points=${listPoints}&expression=${data.function_type.replace("max", "-").replace("min","")}(${data.expression})&constraint_expr1=${data.constraint_expr1}&constraint_type1=${constraintType1}&constraint_value1=${data.constraint_value1}&constraint_expr2=${data.constraint_expr2}&constraint_type2=${constraintType2}&constraint_value2=${data.constraint_value2}&function_type=${data.function_type}&epsilon=${data.epsilon}&x1_initial=${data.x1_initial}&x2_initial=${data.x2_initial}&method=${data.method}`
        console.log("fdffsd",data)
        window.open(`${baseURLSolution}?${data}`, '_blank', 'width=650,height=730');
    }
    const test = () =>{
        setOpenModal(true)
    }

      
    return(
        <>
        <div style={{display: "flex", width: "100%", justifyContent:"space-between"}}>
            <h3>Подробности</h3>
            <div>
                <Button style={{marginRight:"15px"}} variant="contained" className='test' onClick={onSubmit} >Просмотреть график</Button>
                <Button variant="contained"  className='test' onClick={openNewWindowsWithChat} >Просмотреть график ошибок</Button>
            </div>
        </div>
        <div style={{display: "flex", width:"100%", justifyContent: "space-between"}}>
            
            <Card style={{width:"100%"}}>
                <div  style={{display:"flex", flexDirection: "row", justifyContent: "space-around", fontWeight:"800"}}>
                    <p>Итерация</p>
                    <p>x1</p>
                    <p>x2</p>
                </div>
                {solutionData["points"].map((obj, index)=>
                <div style={{display:"flex", flexDirection: "row", justifyContent: "space-around"}}>
                    <p>Итерация {index+1}</p>
                    <p>{obj.x1.toFixed(2)}</p>
                    <p>{obj.x2.toFixed(2)}</p>
                </div>
                )
                }
            </Card>
        </div>
        </>
    )
}