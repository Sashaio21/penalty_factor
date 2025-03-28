import '../App.css';
import './Header.css'
import { Link } from "react-router-dom"
import { Button } from '@mui/material';
import { useEffect, useState } from 'react';
import CloseIcon from '@mui/icons-material/Close';
import {Input} from '@mui/material';

export const  CompareMethod=({solutionData,listDifferences, absoluteDifference, sol, solRepare, relativeDifference})=>{
    const [reference, setReference] = useState()
    const [errorMethod, setErrorMethod] = useState(false)

    const handleChange = (event) => {
        setReference(event.target.value); // Обновляем состояние
        setErrorMethod({
            "solError": relativeDifference(sol.solution.fun, Number(event.target.value)),
            "solRepairError": relativeDifference(solRepare.solution_data.solution.fun, Number(event.target.value))
        })
      };

      const zalupa = {
        "method":"Метод",
        "epsilon":"Точность"
      }
      const zalupa2 = {
        "1":"Квадратичный штраф",
        "3":"Адаптивный штраф"
      }

      function replaceValues(obj) {
        const newObj = { ...obj }; // Создаем копию объекта, чтобы не изменять оригинал
        for (const key in newObj) {
          if (newObj[key] === "1") {
            newObj[key] = "Квадратичный штраф";
          } else if (newObj[key] ==="3") {
            newObj[key] = "Адаптивный штраф";
          }
        }
      
        return newObj;
      }


    function test(){
        console.log("listDifferences ",solutionData)
    }


    return (
        <div style={{display: "flex", flexDirection:"column", padding:"10px",width:"100%"}}>
            <div style={{display: "flex", flexDirection:"row", width:"100%"}}>
                <div style={{display: "flex", flexDirection:"column", width:"100%", justifyContent: "space-between"}}>
                    <div>
                        <p>Значение x1: {solutionData["solution_data"]["solution"]["x1"].toFixed(2)}</p>
                        <p>Значение x2: {solutionData["solution_data"]["solution"]["x2"].toFixed(2)}</p>
                        <p>Значение функции: {solutionData["solution_data"]["solution"]["fun"].toFixed(2)}</p> 
                        <p>Количество итераций: {solutionData["solution_data"]["points"].length}</p>    
                        <p>Время расчёта: {solutionData["solution_data"]["execution_time"].toFixed(2)} cекунд</p>
                        {listDifferences.map((obj)=><p>{zalupa[obj]}: { replaceValues(solutionData)[obj]}</p>)
                        }
                    </div>
                </div>
                <div style={{display: "flex", flexDirection:"column", width:"100%", justifyContent: "space-between"}}>
                    <div style={{ width:"100%"}}>
                            <p>Абсолютная разница: {absoluteDifference(sol.solution.fun, solRepare.solution_data.solution.fun).toFixed(3)}</p>
                            <p>Относительная ошибка: {relativeDifference(sol.solution.fun, solRepare.solution_data.solution.fun).toFixed(3)}%</p>
                                <Input
                                    defaultValue={reference}
                                    onChange={handleChange}
                                    placeholder='Эталонное значение' 
                                    name='reference'

                                />
                            {reference?(
                                <div>
                                    <p>Ошибка нового решения: {errorMethod["solError"].toFixed(3)}%</p>
                                    <p>Ошибка решения «{solutionData.nameSolution}»: {errorMethod["solRepairError"].toFixed(3)}%</p>
                                </div>
                            ):<></>
                            }
                        </div>
                </div>
            </div>
        </div>
    )
}