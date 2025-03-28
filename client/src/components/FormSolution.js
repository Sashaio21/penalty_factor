import { Card, Input, Select, MenuItem, Button, default as Snackbar } from '@mui/material';
import { useForm } from 'react-hook-form';
import { useState } from 'react';
import { useEffect } from 'react';
import axios from '../axios';
import axiosGlobal  from 'axios';
import {Typography} from '@mui/material';
import {Icon} from '@mui/material';
// import Snackbar from '@material-ui/core/Snackbar';
import {CircularProgress} from '@mui/material';
import {Alert} from '@mui/material';


export const FormSolution = ({sol , Snackbar,setDataSolutions,solRepare, setSolRepare,inputSolution ,setSolution, setInputSolution, setOpenModal}) => {
  const [visibleLoader, setVisibleLoader] = useState(false)
  const [open, setOpen] = useState(false);
  const baseUrlSolution = "https://penalty-method-solution.onrender.com"
  // const baseUrlSolution = "http://localhost:5000"
  const  {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm();

    useEffect(()=>{
      console.log("hui", inputSolution)
    },[])


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

    //Функция для замены знаков
    function typesConstraint(){
        return [
        // {"key" :"=", "value" : "="},
        {"key":"> ", "value":"%3E "},
        {"key":"< ", "value":"%3C "},
        {"key":">=", "value":"%3E="},
        {"key":"<=", "value":"%3C="},
        ]
    }

    const handleChange = (e) => {
      const { name, value } = e.target;
      setInputSolution((prev) => ({ ...prev, [name]: value }));
    };

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

    const handleClose = () => {
      setOpen(false)
    };


    const clickk = (data) =>{
      setVisibleLoader(true)
      const datainput = data
      setInputSolution(data)
      console.log("newSOlution",inputSolution)
      data = replacePlusWithPercent2B(data)
      data = `x1_min=-5&x1_max=5&x2_min=-5&x2_max=5&expression=${data.function_type.replace("max", "-").replace("min","")}(${data.expression})&constraint_expr1=${data.constraint_expr1}&constraint_type1=${data.constraint_type1}&constraint_value1=${data.constraint_value1}&constraint_expr2=${data.constraint_expr2}&constraint_type2=${data.constraint_type2}&constraint_value2=${data.constraint_value2}&function_type=${data.function_type}&epsilon=${data.epsilon}&x1_initial=${data.x1_initial}&x2_initial=${data.x2_initial}&method=${data.method}`
      axiosGlobal.get(`${baseUrlSolution}/solution-data?${data}`)
        .then((res)=>{
          setSolution(res.data)
          setVisibleLoader(false)
        }).
        catch((err)=>{
          console.log(err)
          setOpen(true)
        }).finally(()=>{
          setVisibleLoader(false)
        })
      axios.get('/solution/getall')
      .then((data)=>{
        console.log("inputi",datainput)
        console.log("data.data", data.data)
        setDataSolutions(filterObjectsByTemplate(data.data, datainput))
        // setDataSolutions(filterObjectsByTemplate(data.data, inputSolution))
        console.log(filterObjectsByTemplate(data.data, datainput))
      })
      // setVisibleLoader(false)
    }

    return(
      <Card variant='elevation' style={{width:"60%", padding:"10px"}}>
        <form onSubmit={handleSubmit(clickk)}>
          <div>
            <Input
              placeholder='Функция' 
              name='expression'
              defaultValue={inputSolution.expression}
              {...register('expression', { required: "Введите функцию" })}
              style={{border: errors.expression ? "1px solid red" : "1px solid #ccc",}}
            />
            <Select
              id="fruits"
              name="function_type"
              {...register('function_type', { required: "Выберите тип функции" })}
              defaultValue={inputSolution.function_type}
              style={{
                border: errors.function_type ? "1px solid red" : "1px solid #ccc",
                width: "100px",
                height: "30px",
              }}
              MenuProps={{
                PaperProps: {
                  style: {
                    maxHeight: 150, // Максимальная высота выпадающего списка
                    width: 100, // Ширина совпадает с шириной <Select>
                  },
                },
              }}
            >
              <MenuItem value="max">max</MenuItem>
              <MenuItem value="min">min</MenuItem>
            </Select>

          </div>

          <br/>
          <Input
            placeholder='Ограничение 1'
            name='constraint_expr1'
            defaultValue={inputSolution.constraint_expr1}
            {...register('constraint_expr1', { required: "Введите ограничение 1" })}
            style={{border: errors.constraint_expr1 ? "1px solid red" : "1px solid #ccc",}}
          />
          <Select  id="fruits" name="constraint_type1" defaultValue={inputSolution.constraint_type1} style={{border: errors.constraint_type1 ? "1px solid red" : "1px solid #ccc",width: "70px", height: "30px"}} {...register('constraint_type1', { required: "Введите ограничение 1" })}>
            {
              typesConstraint().map((obj) => {return <MenuItem value={obj.value}>{obj.key}</MenuItem>})
            }
          </Select>
          <Input
            placeholder='значение'
            name='constraint_value1'
            style={{border: errors.constraint_value1 ? "1px solid red" : "1px solid #ccc",width: "50px", height: "30px"}}
            type='number'
            defaultValue={inputSolution.constraint_value1}
            {...register('constraint_value1', { required: "Введите ограничение 1" })}

          />
          <br/>
          <br/>
          <Input
            placeholder='Ограничение 2'
            name='constraint_expr2'
            defaultValue={inputSolution.constraint_expr2}
            {...register('constraint_expr2', { required: "Введите ограничение 1" })}
            style={{border: errors.constraint_expr2 ? "1px solid red" : "1px solid #ccc",}}
          />
          <Select  id="fruits" name="constraint_type2" defaultValue={inputSolution.constraint_type2} style={{border: errors.constraint_type2 ? "1px solid red" : "1px solid #ccc",width: "70px", height: "30px"}} {...register('constraint_type2', { required: "Введите ограничение 1" })}>
            {
              typesConstraint().map((obj) => {return <MenuItem value={obj.value}>{obj.key}</MenuItem>})
            }
          </Select>
          <Input
            placeholder='значение'
            name='constraint_value2'
            style={{border: errors.constraint_value2 ? "1px solid red" : "1px solid #ccc",width: "50px", height: "30px"}}
            type='number'
            defaultValue={inputSolution.constraint_value2}
            {...register('constraint_value2', { required: "Введите ограничение 1" })}

          />

          <div style={{display: "flex"}}>
            <p style={{marginRight: "20px"}}>Начальные точки</p>
            <div style={{marginTop: "15px"}}>
              <Input
                placeholder='x1'
                defaultValue={inputSolution.x1_initial}
                name='x1_initial'
                style={{border: errors.x1_initial ? "1px solid red" : "1px solid #ccc",width: "50px", height: "30px"}}
                type='number'
                {...register('x1_initial', { required: "Введите ограничение 1" })} 
              />
              <Input
                placeholder='x2'
                defaultValue={inputSolution.x2_initial}
                name='x2_initial'
                style={{border: errors.x2_initial ? "1px solid red" : "1px solid #ccc",width: "50px", height: "30px"}}
                type='number'
                {...register('x2_initial', { required: "Введите ограничение 1" })}  
              />
            </div>
          </div>
          <div style={{display: "flex"}}>
            <p style={{marginRight: "20px"}}>Точность</p>
            <Input
              placeholder="0.0"
              defaultValue={inputSolution.epsilon}
              name="epsilon"
              style={{
                border: errors.epsilon ? "1px solid red" : "1px solid #ccc",
                width: "70px",
                height: "30px",
                marginTop: "15px",
              }}
              type="string"
              {...register("epsilon", {
                required: "Введите значение",
                validate: (value) => {
                  if (value.includes(",")) {
                    return "Используйте точку вместо запятой";
                  }
                  if (!/^\d+(\.\d+)?$/.test(value)) {
                    return "Введите корректное число (например, 0.5)";
                  }
                  return true;
                },
              })}
            />
            {errors.epsilon && <p style={{marginLeft:"10px" , color: "red" }}>{errors.epsilon.message}</p>}

          </div>
          <div style={{display: "flex"}}>
            <p style={{marginRight: "5px"}}>Выбор метода</p>
            <Select id="fruits" name="method" defaultValue={inputSolution.method}  {...register('method', { required: "Введите ограничение 1" })} style={{border: errors.method ? "1px solid red" : "1px solid #ccc",width: "220px", marginTop: "15px",height: "30px"}}>
              <MenuItem value="1">Квадратичный штраф</MenuItem>
              {/* <MenuItem value="2">Method 2</MenuItem> */}
              <MenuItem value="3">Адаптивный штраф</MenuItem>
            </Select>
          </div>    
          <br/>
          <div style={{display: "flex", }}>
            <Button variant="contained" className='test' type='submit' >Рассчитать</Button>
            {visibleLoader ? (
              <CircularProgress 
              style={{
                marginLeft:"24%"
              }}
            />
            ): <></>}

          </div>
          {Object.keys(errors).length > 0 && (
          <Typography style={{ color: "red", marginTop: "10px" }}>
            Заполните все поля
          </Typography>
        )}
        </form>
        
        <Snackbar open={open} autoHideDuration={6000} onClose={handleClose}>
        <Alert
          onClose={handleClose}
          severity="error"
          variant="filled"
          sx={{ width: '100%' }}
        >
          Произошла ошибка на сервере. Обновите страницу и попробуйте ещё раз
        </Alert>
      </Snackbar>
      </Card>
    )
}