import '../App.css';
import { Routes, Route, Link, useParams } from 'react-router-dom';
import { useForm } from 'react-hook-form';
import { useEffect, useState, useLayoutEffect } from 'react';
import { Card, Input, Select, MenuItem, Button } from '@mui/material';
import { CompSolution } from '../components/CompSolution';
import { FormSolution } from '../components/FormSolution';
import axios from '../axios';
import {Snackbar} from '@mui/material';
import {CardHeader} from "@mui/material"
import { MainResult } from '../components/MainResult';
import {Modal} from '@mui/material';
import moment from 'moment';
import {CompareMethod} from '../components/CompareMethod';
// import {IconButton} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import {Typography} from '@mui/material';
import {CircularProgress} from '@mui/material';
import {CardActionArea} from '@mui/material';


function Solution() {
  const [sol, setSol] = useState()
  const [solRepare, setSolRepare] = useState()
  const [open, setOpen] = useState(false);
  const [dataSolutions, setDataSolutions] = useState(["", "", "", "", "", "", "", ""])

  const [inputValue, setInputValue] = useState("Новое решение"); // Начальное значение
  const [errNameSolution, setErrNameSolution] = useState(false)
  const [openModal, setOpenModal] = useState(false)
  const handleOpen = () => setOpenModal(true)
  const handleCloseModal = () => {setOpenModal(false)}
  const [listDifferences, setListDifferences] = useState([])
  const [visibleForm, setVisibleForm] = useState(false)


  const [inputSolution, setInputSolution] = useState("")
  const {idSolution} = useParams();

  const testr = () =>{
    // axios.get(`/solution/get/${idSolution}`)
    // .then((data)=>{
    //   setInputSolution(data.data)
    // })
    console.log("retrefgdgdfgdf" ,inputSolution)
  }

  function filterObjectsByTemplate(objects, template) {
    console.log("dfdsfdsf",template)
    return objects.filter(obj => 
        obj.expression == template.expression &&
        obj.constraint_expr1 == template.constraint_expr1 &&
        obj.constraint_type1 == template.constraint_type1 &&
        obj.constraint_value1 == template.constraint_value1 &&
        obj.constraint_expr2 == template.constraint_expr2 &&
        obj.constraint_type2 == template.constraint_type2 &&
        obj.constraint_value2 == template.constraint_value2
    );
}

  const getSolutionsForRepare =()=>{
    if (window.localStorage.getItem('token')) {
      axios.get('/solution/getall')
      .then((data)=>{
        setDataSolutions(filterObjectsByTemplate(data.data, inputSolution))
        console.log("fffffffffffffffffff",data.data)
        console.log(inputSolution)
      })
    }

  }

  useEffect(()=>{
    getSolutionsForRepare()
    if (idSolution!="newSolution") {
      axios.get(`/solution/get/${idSolution}`)
      .then((data)=>{
        console.log("idSolution", idSolution)
        setInputSolution(data.data)
        setSol(data.data.solution_data)
        console.log("sol", data.data)
      }) 
    }
    else{
      setVisibleForm(true)
    }
    // console.log("retrefgdgdfgdf" ,inputSolution)
  },[])


  function findDifferences(obj1, obj2) {
    const differences = [];
    if (obj1.method !== obj2.method) {
      differences.push("method");
    }
    if (Number(obj1.epsilon) !== Number(obj2.epsilon)) {
      differences.push("epsilon");
    }
    return differences;
  }
  
  const selectSolutionForRepare = (id)=>{
    if (window.localStorage.getItem('token')) {
      axios.get('/solution/getall')
      .then((data)=>{
        setDataSolutions(filterObjectsByTemplate(data.data, inputSolution))
        console.log(dataSolutions)
      })
    }

    setListDifferences(findDifferences(inputSolution, dataSolutions[id]))
    console.log("differences  ",findDifferences(inputSolution, dataSolutions[id]))
    setSolRepare(dataSolutions[id])
    handleCloseModal(false)
  }

  const handleClose = () => {
    setOpen(false)
  };



    



    // Функция для вычисления абсолютной разницы
    function absoluteDifference(value1, value2) {
      return Math.abs(value1 - value2);
    }

    // Функция для вычисления относительной разницы
    function relativeDifference(value1, value2) {
      const absoluteDiff = absoluteDifference(value1, value2);
      console.log("value1 ",value1)
      console.log("value2 ",value2)
      const average = (Math.abs(value1) + Math.abs(value2)) / 2;
      return absoluteDiff / average;
    }

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




  const methods = {
    "1":"Квадратичный штраф",
    "3":"Адаптивный штраф"
  }


  const handleChange = (event) => {
    setInputValue(event.target.value); // Обновляем состояние
  };

  //Сохранение решения
  const saveSolution = () =>{
    console.log("solution ",{...sol})
    var dataSolut = {}
    if (inputValue=="") {
      dataSolut = {
        nameSolution: "Новое решение",
        ...inputSolution,
        "solution_data" : {...sol}
      }
    } else{
      dataSolut = {
        nameSolution: inputValue,
        ...inputSolution,
        "solution_data" : {...sol}
      }
    }

    if (window.localStorage.getItem('token')) {
      axios.get('/solution/getall')
      .then((data)=>{
        setDataSolutions(filterObjectsByTemplate(data.data, inputSolution))
        console.log("fffffffffffffffffff",data.data)
        console.log(inputSolution)
      })
    }
    
    axios.post('/solution/add', dataSolut)
    .then((res)=>{
      setOpen(true)
      console.log(res)
      setErrNameSolution(false)
    })
    .catch((err)=>{
      // console.log(err.response.data.error)
      setErrNameSolution(err.response.data.error)
    })
  }
  
  return (
    <div>
      <div style={{display: "flex", justifyContent: "space-between"}}>
        <h2 style={{height:"40px"}}>Входные параметры</h2>
        {idSolution=="newSolution" ? (
          <div style={{height:"100%", marginTop:"20px"}}>
            <Input
              defaultValue={inputValue}
              placeholder='Название решения'
              onChange={handleChange} 
              style={{
                marginRight:"20px",
                border: !errNameSolution.expression == "" ? "1px solid red" : "1px solid #ccc",
              }}
            ></Input>
            {sol ? (
              <Button onClick={()=>saveSolution() } variant="contained">Сохранить результат</Button>
            ):<Button disabled >Сохранить результат</Button>
            }
            {errNameSolution ? (
              <Typography style={{ color: "red", marginTop: "0px", fontSize:"15px" }}>
              {errNameSolution}
            </Typography>
            ):<></>}
            
          </div>  
        ):<i style={{
          marginTop:"20px",
          fontSize:"18px"
        }}
        >{inputSolution.nameSolution}</i>}      
      </div>
      <div style={{display: "flex", width:"100%", justifyContent: "space-between"}}>
        {visibleForm || inputSolution!="" ? (
          <FormSolution setOpenModal={setOpenModal} Snackbar={Snackbar} setDataSolutions={setDataSolutions} solRepare={solRepare} setSolRepare={setSolRepare} sol={sol} inputSolution={inputSolution} setSolution={setSol} setInputSolution={setInputSolution}></FormSolution>
        ): <></>}
        <Card style={{width:"50%", marginLeft:"10px", height:"100%", padding:"20px"}}>
          <Typography>
            Все переменные выражаются через x
          </Typography>
          <Typography>
            Все математические операции выражаются через общепринятые символы +, -, *, /, **
          </Typography>
          <i>Пример:</i>
          <Typography>
            Функция: x1**2+x2**2-20*x1-30*x2
          </Typography>
          <Typography>
            Ограничение 1: 2*x1+3*x2-13
          </Typography>
          <Typography>
            Ограничение 2: 2*x1+x2-10
          </Typography>
        </Card>
      </div>
      <div style={{display: "flex", flexDirection:"row", width: "100%", justifyContent:"space-between"}}>
        {sol?(<MainResult getSolutionsForRepare={getSolutionsForRepare} setOpenModal={setOpenModal} solutionData={sol}></MainResult>):<></>}
        {solRepare? (
          <div style={{
              display: "flex", 
              flexDirection:"column", 
              width:"70%"}}>
            <div style={{display:"flex", width:"100%", justifyContent:"space-between"}}>
              <h3 >Cравнение</h3>
              <div onClick={()=>setSolRepare(false)} style={{display: "flex", cursor: "pointer",flexDirection:"row", alignSelf:"flex-end", justifyContent:"space-between"}}>
                  <i style={{paddingRight:"10px"}}>{solRepare.nameSolution}</i>
                  <CloseIcon style={{color:"red"}}></CloseIcon>
              </div>
            </div>
            <Card style={{display: "flex", flexDirection:"row", width:"100%", justifyContent:"space-between"}}>
            <CompareMethod listDifferences={listDifferences} absoluteDifference={absoluteDifference} relativeDifference={relativeDifference} sol={sol} solRepare={solRepare} setSolRepare={setSolRepare} solutionData={solRepare}></CompareMethod>
          </Card>
          </div>
        ):<></>
        }

      </div>
      <br></br>
      
      {sol ? (<CompSolution setOpenModal={setOpenModal} solutionData={sol} inputSolution={inputSolution}/>):<></>}
      <Modal
        open={openModal}
        onClose={handleCloseModal}
      >
        {dataSolutions != 0 ? (
          <Card 
          style={{
          position: "absolute",
          top: "10%",
          left: "50%",
          transform: "translate(-50%, 0)",
          width: "25%",
          maxHeight: "80vh", // Ограничиваем высоту модального окна
          overflowY: "auto", // Включаем вертикальную прокрутку
          padding: "20px",
        }}
        >
              {dataSolutions.map((obj, index)=>
              <CardActionArea>
                <div style={{width: "100%"}} onClick={()=>selectSolutionForRepare(index)}>
                <div style={{display:"flex",flexDirection:"row", width:"100%", justifyContent:"space-between"}}>
                  <h3 style={{marginBottom: "15px"}}>{obj.nameSolution}</h3>
                  <h3 style={{marginBottom: "15px"}}>{ moment(obj.createdAt).format('DD.MM.YYYY HH:mm')}</h3>
                </div>
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
                <hr></hr>
              </div>
              </CardActionArea>
              )
              }
        </Card>
        ): <Card 
        style={{
        position: "absolute",
        top: "10%",
        left: "50%",
        transform: "translate(-50%, 0)",
        width: "25%",
        maxHeight: "80vh", // Ограничиваем высоту модального окна
        overflowY: "auto", // Включаем вертикальную прокрутку
        padding: "20px",
      }}
      >
        <i style={{
          marginTop:"20px",
          fontSize:"18px"
        }}
        >Нет подходящий решений для сравнения</i>
      </Card>}
      </Modal>
        <Snackbar
          open={open}
          autoHideDuration={5000}
          onClose={handleClose}
          message="Решение сохранено"
          action={<Button color="inherit" size="small" onClick={handleClose}>Закрыть</Button>}
        />
    </div>
  );
}

export default Solution;