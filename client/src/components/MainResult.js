import {  Button, Card } from "@mui/material"


export const MainResult = ({solutionData, getSolutionsForRepare, setOpenModal}) => {

    const getsDataSolutions = ()=>{
        getSolutionsForRepare()
        setOpenModal(true)
    }

    return(
        <div style={{height:"100%"}}>
            <h3>Результат</h3>
            <div style={{display: "flex", flexDirection:"column", justifyContent: "space-between"}}>
                <Card style={{padding:"15px"}}> 
                    <p>Значение x1: {solutionData["solution"]["x1"].toFixed(2)}</p>
                    <p>Значение x2: {solutionData["solution"]["x2"].toFixed(2)}</p>
                    <p>Значение функции: {solutionData["solution"]["fun"].toFixed(2)}</p> 
                    <p>Количество итераций: {solutionData["points"].length}</p>    
                    <p>Время расчёта: {solutionData["execution_time"].toFixed(2)} cекунд</p>  
                </Card>
                <Button  variant="contained" onClick={()=>{getsDataSolutions()}}>Сравнить</Button>
            </div>
        </div>
    )
}