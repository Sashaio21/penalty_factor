import '../App.css';
import './styleMain.css'
import { Routes, Route, Link } from 'react-router-dom';
import { useForm } from 'react-hook-form';
import { useEffect, useState } from 'react';
import { Card, Input, Select, MenuItem, Button, Alert} from '@mui/material';
import { CompSolution } from '../components/CompSolution';
import { FormSolution } from '../components/FormSolution';
import axios from '../axios';
import moment from 'moment';
import { Navigate } from 'react-router-dom';
import {Snackbar} from '@mui/material';
import {Skeleton} from '@mui/material';


function Home() {
  const [sol, setSol] = useState()
  const [inputSolution, setInputSolution] = useState("")
  const [dataSolutions, setDataSolutions] = useState(["", "", "", "", "",""])
  const [errorListSol, setErrorListSol] = useState(false)
  const [openErr, setOpenErr] = useState(false);


  
  useEffect(()=>{
      if (window.localStorage.getItem('token')) {
        axios.get('/solution/getall')
        .then((data)=>{
          setDataSolutions(data.data)
          console.log(data.data)
        }).catch((error)=>{
          setOpenErr(true)
        })
      }
  },[])


  // функция заменяющая знаки неравенств
  function replacSigns(userSign){
    const signs = {
      "%3E ":"> ",
      "%3C ":"< ",
      "%3E=":">=",
      "%3C=":"<="
    }
    return signs[userSign]
  }

  const handleCloseErr = () => {
    setOpenErr(false)
  };
  
  const methods = {
    "1":"Квадратичный штраф",
    "3":"Адаптивный штраф"
  }

  if (!window.localStorage.getItem('token')) {
    return <Navigate to={'/login'}/> 
  }


  return (
    <div>
      <Link to={'/solution/newSolution'} style={{alignSelf: "flex-end"}}><Button  variant="contained">Создать расчёт</Button></Link>
      {dataSolutions[0] === "" ? (
        <div className='storySolution'>
          {dataSolutions.map((obj) => (
            <div style={{width:"100%" , height:"385px"}}>
              <Skeleton style={{width:"100%",height:"550px",marginTop:"-90px", padding:"0px"}} key={obj._id}></Skeleton>
            </div>

        ))}
        </div>
      ) : (
        dataSolutions.map((obj)=>
          <Card className='storySolution'>
            <div style={{display:"flex",flexDirection:"row", width:"100%", justifyContent:"space-between"}}>
              <h4 style={{marginBottom: "15px"}}>{obj.nameSolution}</h4>
              <h4 style={{marginBottom: "15px"}}>{ moment(obj.createdAt).format('DD.MM.YYYY  HH:mm')}</h4>
            </div>

            <div style={{display: 'flex', justifyContent:"space-between"}}>
              <div style={{width: "50%"}}>
                <div style={{display: "flex", justifyContent:"space-between", marginBottom: "15px"}}>
                  <div>
                    Целевая функция: 
                  </div>
                  <div>
                    {obj.expression}
                  </div>
                </div>

                <div style={{display: "flex", justifyContent:"space-between", marginBottom: "15px"}}>
                  <div>
                  Ограничение 1: 
                  </div>
                  <div>
                    {obj.constraint_expr1}{replacSigns(obj.constraint_type1)}{obj.constraint_value1}
                  </div>
                </div>

                <div style={{display: "flex", justifyContent:"space-between", marginBottom: "15px"}}>
                  <div>
                    Ограничение 2:  
                  </div>
                  <div>
                    {obj.constraint_expr2}{replacSigns(obj.constraint_type2)}{obj.constraint_value2}
                  </div>
                </div>
                <div style={{display: "flex", justifyContent:"space-between", marginBottom: "15px"}}>
                  <div>
                    Вид штрафа:  
                  </div>
                  <div>
                    {methods[obj.method]}
                  </div>
                </div>
              </div>
              <div style={{marginLeft: "50px", width: "50%"}}>

                <div style={{display: "flex", flexDirection: "column",justifyContent:"space-between", marginBottom: "15px"}}>
                  <div>
                    Оптимальные точки: 
                  </div>
                  <div>
                    <div className='textDefault'>x1: {obj.solution_data.solution.x1.toFixed(2)}</div>
                    <div className='textDefault'>x2: {obj.solution_data.solution.x2.toFixed(2)}</div>
                  </div>
                </div>                
                <div className='textDefault'>Решение функции: {obj.solution_data.solution.fun.toFixed(2)}</div>           
              </div>
            </div>
            <Link to={`/solution/${obj._id}`} style={{ alignSelf: "flex-end"}}><Button style={{width: "150px", marginBottom: "10px"}} >Подробнее</Button></Link>
          </Card>
          
      )
      )}

    
      <Snackbar open={openErr} autoHideDuration={6000} onClose={handleCloseErr}>
        <Alert
          onClose={handleCloseErr}
          severity="error"
          variant="filled"
          sx={{ width: '100%' }}
        >
          Произошла ошибка на сервере{"\n"}Перезагрузите страницу
        </Alert>
      </Snackbar>
    </div>
  );
}

export default Home;